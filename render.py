"""
Stage 3: tracks in, annotated video out.

    clip.mp4 + tracks.json  --PIL--> annotated.mp4

Nothing here looks at the image. Every position was produced by a model and
turned into an identity by geometry; this file only draws. That is the line the
brief draws and it is worth being able to point at: Pillow reads no pixel it did
not itself write.

Frames stream through two ffmpeg pipes rather than landing on disk. 900 frames of
1280x720 RGB is 2.5 GB, which is fine to move through memory a frame at a time
and unpleasant to write out as 900 files and read back.

THE MARKER SPEC (the user's, D11):
  - flat ellipse under the feet, in the team's colour
  - jersey number above the head, same colour
  - two numbers that crowd each other both fade, and fade further the closer
    they get, so an unreadable overlap degrades into transparency rather than
    into a smear
  - the player on the ball is marked differently again
  - the ball gets its own marker

TEAM COLOUR RESOLUTION:
  Use each team's own kit colour. If the two kits are too close to tell apart at
  a glance, the second team switches to its accent colour — provided that accent
  is itself distinguishable. Failing that, fall back to the hue opposite the
  first team's colour, which is distinguishable by construction.

  "Too close" is measured as CIE76 dE in Lab space, not RGB distance. RGB
  distance disagrees with human vision badly enough to matter here: navy and
  black are far apart in RGB and nearly identical on a floodlit pitch.

    uv run render.py outputs/tracks/allstars_fr_eng__dev__tracks.json
    uv run render.py <tracks.json> --clip clips/other.mp4 --out outputs/x.mp4
"""

import argparse
import colorsys
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = Path("outputs/videos")

# Kit vocabulary. The prompt asks for "one ordinary word", and these are the
# words that come back. Unknown words fall through to a neutral grey and are
# reported, so a colour we have not seen becomes a visible gap rather than a
# silent mis-render.
NAMED = {
    "red": (206, 32, 41), "crimson": (190, 30, 55), "maroon": (120, 24, 40),
    "claret": (124, 32, 56), "burgundy": (110, 28, 46),
    "blue": (28, 68, 190), "navy": (18, 30, 84), "royal": (32, 74, 200),
    "sky": (110, 180, 235), "skyblue": (110, 180, 235),
    "lightblue": (140, 195, 240), "darkblue": (20, 38, 110),
    "teal": (24, 130, 132), "turquoise": (48, 190, 180), "cyan": (40, 210, 220),
    "white": (245, 245, 245), "cream": (238, 232, 205), "beige": (226, 214, 182),
    "black": (24, 24, 26), "grey": (128, 128, 132), "gray": (128, 128, 132),
    "silver": (188, 190, 195),
    "green": (32, 140, 62), "lime": (110, 200, 60), "darkgreen": (20, 78, 40),
    "yellow": (240, 214, 48), "gold": (218, 176, 52), "amber": (236, 178, 40),
    "orange": (236, 126, 34), "pink": (232, 130, 170), "purple": (118, 56, 160),
    "violet": (140, 84, 200), "brown": (110, 72, 46),
}
FALLBACK = (150, 150, 150)

# CIE76 dE. ~2.3 is "just noticeable"; 30 is comfortably "different colour".
# We want at-a-glance separation on moving video, not a lab match, so the bar is
# set high deliberately.
TOO_CLOSE_DE = 30.0

ELLIPSE_WIDTH_MUL = 1.9      # of the player's box width
ELLIPSE_ASPECT = 0.36        # height as a fraction of ellipse width
ELLIPSE_MIN_W = 14           # px, so distant players still get a visible mark

# Smaller and lighter than the first pass, which sat on the footage rather than
# in it. A condensed face reads at small sizes and takes less horizontal room,
# which also means fewer neighbours trip the crowding fade.
NUM_MIN_PX, NUM_MAX_PX = 9, 21
NUM_SIZE_MUL = 0.30          # of box height (was 0.42)
NUM_GAP_MUL = 0.26           # number sits this far above the head, in box heights
FONT_STACK = ("bahnschrift.ttf", "seguisb.ttf", "tahomabd.ttf",
              "DejaVuSansCondensed-Bold.ttf", "arialbd.ttf")

# An invented id is not a jersey number and must not look like one. Read numbers
# are drawn solid; invented ones are drawn hollow — outline only — so a viewer
# can tell at a glance which labels are evidence and which are bookkeeping.
# Letters carry the distinction on their own — jerseys have digits, so anything
# alphabetic is obviously not a read number. Shrinking and fading them ON TOP of
# that made them invisible rather than merely distinguishable, which is the
# opposite of the point. Same size, same weight, same colour as a read number.
INVENTED_SIZE_MUL = 1.0
INVENTED_ALPHA_MUL = 1.0
INVENTED_DESAT = 0.0

FADE_MIN_ALPHA = 0.22        # a fully crowded number never vanishes completely
FADE_SPAN_MUL = 2.4          # crowding is measured in multiples of glyph width

# Attention fade. Seventeen numbers on screen is not seventeen times as useful as
# five — past a point each extra label costs the viewer more than it tells them.
# The ball is where the eye already is, so relevance is distance from the ball:
# full opacity nearby, fading to nothing beyond FAR, where the marker ellipse
# alone carries team and position.
#
# Both radii scale with how crowded the frame is. A packed penalty box needs a
# tighter radius than a spread-out midfield, or the rule does nothing exactly
# when it is most needed.
# Rewritten. The first version multiplied two independent fades — a crowding
# fade and a distance fade whose radius shrank with player count — and on a real
# 17-player frame they compounded into "everything is permanently half
# transparent". Both were doing their job; the product was the bug.
#
# Now it is a RANK, which is what the intent was all along: the N players
# closest to the ball get their number, the next few fade out, the rest get
# nothing. A rank cannot compound, and the number of visible labels is the thing
# actually being controlled rather than a side effect of two radii.
LABELS_FULL = 6              # this many nearest the ball, at full opacity
LABELS_FADE = 4              # then this many fading to nothing
CROWD_MIN_ALPHA = 0.45       # floor for the overlap fade (was 0.22)
CROWD_SPAN_MUL = 1.15        # only genuinely overlapping labels (was 2.4)

BALL_R_MIN = 5
ON_BALL_RING_MUL = 1.35      # the on-ball halo, relative to the team ellipse


# ---------------------------------------------------------------- colour

def srgb_to_lab(rgb):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(float(v)) for v in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.00000
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a, b):
    la, aa, ba = srgb_to_lab(a)
    lb, ab, bb = srgb_to_lab(b)
    return math.sqrt((la - lb) ** 2 + (aa - ab) ** 2 + (ba - bb) ** 2)


def named(word):
    return NAMED.get((word or "").strip().lower())


def opposite(rgb):
    """The hue 180 degrees away, kept bright enough to read on grass."""
    h, s, v = colorsys.rgb_to_hsv(*[c / 255 for c in rgb])
    r, g, b = colorsys.hsv_to_rgb((h + 0.5) % 1.0, max(s, 0.75), max(v, 0.85))
    return tuple(int(round(c * 255)) for c in (r, g, b))


def resolve_team_colours(kit_a, kit_b, accents):
    """Two colours that a viewer can tell apart at a glance. Returns
    (colour_a, colour_b, note) where note records which rule fired."""
    ca = named(kit_a) or FALLBACK
    cb = named(kit_b) or FALLBACK
    if kit_b is None:
        return ca, opposite(ca), "single kit — second colour is the opposite hue"
    if delta_e(ca, cb) >= TOO_CLOSE_DE:
        return ca, cb, "kit colours are distinguishable; used as-is"

    acc_b = named(accents.get((kit_b or "").lower()))
    acc_a = named(accents.get((kit_a or "").lower()))
    # The accent has to clear BOTH the colour actually being drawn for team A and
    # A's own accent. The spec named the second check; the first matters more,
    # because A's drawn colour is what B's marker sits next to on screen.
    if acc_b and delta_e(acc_b, ca) >= TOO_CLOSE_DE \
            and (acc_a is None or delta_e(acc_b, acc_a) >= TOO_CLOSE_DE):
        return ca, acc_b, f"kits too close (dE {delta_e(ca, cb):.0f}); " \
                          f"used team B's accent '{accents.get(kit_b)}'"
    return ca, opposite(ca), (f"kits too close (dE {delta_e(ca, cb):.0f}) and "
                              f"accent unusable; used the opposite hue")


# ---------------------------------------------------------------- drawing

def load_font(px):
    for name in FONT_STACK:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(px)
    except TypeError:
        return ImageFont.load_default()


def probe_size(clip: Path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0:s=x", str(clip)],
        capture_output=True, text=True)
    parts = r.stdout.strip().split("x")
    w, h = int(parts[0]), int(parts[1])
    num, den = (parts[2].split("/") + ["1"])[:2]
    return w, h, float(num) / float(den)


def paste_glow(base, cx, cy, rx, ry, fill, blur):
    """Blur a soft ellipse onto `base`, working only on the region it touches.

    A Gaussian blur costs time proportional to the area it covers, so blurring a
    full 1280x720 layer to light up a 40-pixel circle wastes ~99.8% of the work.
    Crop a box big enough to hold the shape plus three blur radii of falloff,
    blur that, and composite it back.
    """
    pad = int(blur * 3) + 4
    x0 = max(0, int(cx - rx - pad)); y0 = max(0, int(cy - ry - pad))
    x1 = min(base.width, int(cx + rx + pad)); y1 = min(base.height, int(cy + ry + pad))
    if x1 <= x0 or y1 <= y0:
        return base
    layer = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse(
        [cx - rx - x0, cy - ry - y0, cx + rx - x0, cy + ry - y0], fill=fill)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(layer, (x0, y0))
    return base


def number_alphas(labels):
    """Crowding fade.

    Two numbers drawn on top of each other are worse than one number: the reader
    cannot resolve either, and the picture looks broken rather than busy. Fading
    both, in proportion to how close they are, turns an unreadable collision into
    an honest visual signal that two players are overlapping.

    `labels` is [(x_px, y_px, glyph_w_px), ...]; returns one alpha per label.
    """
    n = len(labels)
    alphas = [1.0] * n
    for i in range(n):
        xi, yi, wi = labels[i]
        for j in range(i + 1, n):
            xj, yj, wj = labels[j]
            span = CROWD_SPAN_MUL * max(wi, wj)
            d = math.hypot(xi - xj, yi - yj)
            if d >= span:
                continue
            a = CROWD_MIN_ALPHA + (1.0 - CROWD_MIN_ALPHA) * (d / span)
            alphas[i] = min(alphas[i], a)
            alphas[j] = min(alphas[j], a)
    return alphas


def draw_frame(base, entry, colours, W, H, font_cache, ball_fade=False,
               boxes=False):
    """Draw one frame's annotations onto `base` (an RGB Image). Returns it."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    players = entry["players"]
    labels, metas, glow_marks = [], [], []

    for p in players:
        col = colours.get(p.get("team")) or FALLBACK
        fx, fy = p["x"] * W, p["y"] * H
        bw, bh = p["w"] * W, p["h"] * H
        ew = max(ELLIPSE_MIN_W, bw * ELLIPSE_WIDTH_MUL)
        eh = ew * ELLIPSE_ASPECT

        if boxes:
            # DIAGNOSTIC MODE, not a deliverable. Every model returns a box
            # aspect of ~2.2 where a standing player is ~3.5 — 37% too short —
            # but "too short" does not say WHICH edge is wrong, and the two have
            # opposite fixes. If the TOP is right and the BOTTOM rides high, the
            # model is boxing torsos and the foot point (y) is wrong, so the
            # markers sit above the feet. If the BOTTOM is right and the TOP is
            # low, the foot point is already correct and asking for it directly
            # (HANDOFF item 6) buys nothing.
            #
            # The two edges are therefore coloured differently and named in the
            # legend, so the answer is "cyan sits at the head / magenta sits at
            # the boots" rather than a judgement about the box as a whole.
            x0, y0, x1, y1 = fx - bw / 2, fy - bh, fx + bw / 2, fy
            d.rectangle([x0, y0, x1, y1], outline=col + (200,), width=1)
            d.line([x0, y0, x1, y0], fill=(0, 229, 255, 255), width=2)   # top
            d.line([x0, y1, x1, y1], fill=(255, 0, 200, 255), width=2)   # bottom
        elif p.get("on_ball"):
            # "Differently again" as a soft glow on their own marker rather than
            # a hard white ring around it. The ring was a second high-contrast
            # edge competing with the ellipse it surrounded; a glow reads as
            # emphasis without adding another line to the picture.
            glow_marks.append((fx, fy, ew, eh, col))
            d.ellipse([fx - ew / 2, fy - eh / 2, fx + ew / 2, fy + eh / 2],
                      fill=col + (120,), outline=(255, 255, 255, 245), width=3)
        else:
            d.ellipse([fx - ew / 2, fy - eh / 2, fx + ew / 2, fy + eh / 2],
                      fill=col + (55,), outline=col + (225,), width=2)

        read = bool(p.get("read", True))
        size = int(max(NUM_MIN_PX, min(NUM_MAX_PX, bh * NUM_SIZE_MUL
                                       * (1.0 if read else INVENTED_SIZE_MUL))))
        if size not in font_cache:
            font_cache[size] = load_font(size)
        font = font_cache[size]
        text = str(p.get("label") or p["id"]).split("·")[-1]
        tw = d.textlength(text, font=font)
        ty = fy - bh - bh * NUM_GAP_MUL - size
        labels.append((fx, ty, tw))
        metas.append((text, font, col, fx, ty, read))

    # Relevance by RANK: order everyone by distance to the ball and show the
    # nearest few. Independent of frame size, player count and pitch geometry,
    # and it caps the label count directly instead of hoping two radii produce
    # the right number between them.
    ball = entry.get("ball")
    rel = [1.0] * len(players)
    if ball_fade and ball and len(players) > LABELS_FULL:
        order = sorted(range(len(players)),
                       key=lambda i: math.hypot(players[i]["x"] - ball["x"],
                                                players[i]["y"] - ball["y"]))
        for rank, i in enumerate(order):
            if players[i].get("on_ball"):
                rel[i] = 1.0        # never fade the player in possession
            elif rank < LABELS_FULL:
                rel[i] = 1.0
            elif rank < LABELS_FULL + LABELS_FADE:
                rel[i] = 1.0 - (rank - LABELS_FULL + 1) / (LABELS_FADE + 1)
            else:
                rel[i] = 0.0

    for (text, font, col, tx, ty, read), crowd, r in zip(
            metas, number_alphas(labels), rel):
        alpha = min(crowd, r) * (1.0 if read else INVENTED_ALPHA_MUL)
        if alpha <= 0.02:
            continue                # fully faded — skip the draw entirely
        a = int(round(255 * alpha))
        if read:
            d.text((tx, ty), text, font=font, fill=col + (a,), anchor="mt",
                   stroke_width=2, stroke_fill=(0, 0, 0, int(a * 0.8)))
        else:
            d.text((tx, ty), text, font=font, fill=col + (a,), anchor="mt",
                   stroke_width=2, stroke_fill=(0, 0, 0, int(a * 0.8)))

    # Glows are blurred on a small CROP, not on a full-frame layer.
    #
    # The first version built a 1280x720 RGBA layer, drew a 40-pixel ellipse on
    # it, Gaussian-blurred the entire thing and alpha-composited it over the
    # frame — once for the possession glow and again for the ball. Two
    # full-frame blurs and three full-frame composites per frame, to light up
    # two small circles. That was most of the 81ms per frame.
    out = base.convert("RGBA")
    for fx, fy, ew, eh, col in glow_marks:
        out = paste_glow(out, fx, fy, ew * 0.85, eh * 0.85, col + (150,), eh * 0.9)
    out = Image.alpha_composite(out, overlay)

    if entry.get("ball"):
        # Two hard concentric rings read as an eyesore: they are high-contrast,
        # they fight the ball itself for attention, and at this size the gap
        # between them shimmers frame to frame. Replaced with a soft glow that
        # sits BEHIND a single thin ring — the glow says "look here" without any
        # hard edges, and the ring stays crisp enough to locate precisely.
        b = entry["ball"]
        bx, by = b["x"] * W, b["y"] * H
        # The marker used to be r = max(BALL_R_MIN, W * 0.006) — a flat 7.7px at
        # 720p, identical in every frame of every clip. The model reports the
        # ball's own w and h on every detection and both were being discarded,
        # so a ball 20 metres away and a ball filling the goalmouth drew the same
        # dot, and a basketball drew smaller than it actually is:
        #
        #   clip         model's ball width      drawn as
        #   allstars       7.7 - 17.9 px          7.7 px
        #   basketball    15.4 - 33.3 px          7.7 px
        #   cuts           7.7 - 129.3 px         7.7 px
        #
        # Mean of w and h, because the box is occasionally flat when the ball is
        # clipped by a player; the mean degrades more gracefully than either
        # alone. BALL_R_MIN still floors it so a distant ball stays visible, and
        # the cap stops a mis-sized box (the cuts clip has a 0.101 outlier)
        # painting a dinner plate over the pitch.
        r_model = (b.get("w", 0.0) * W + b.get("h", 0.0) * H) / 4.0
        r = min(max(BALL_R_MIN, r_model), W * 0.05)
        out = paste_glow(out, bx, by, r * 2.4, r * 2.4,
                         (255, 226, 92, 62), r * 1.4)
        ImageDraw.Draw(out).ellipse(
            [bx - r, by - r, bx + r, by + r],
            outline=(255, 255, 255, 240), width=max(1, int(r * 0.45)))

    return out.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--crf", type=int, default=20)
    # OFF by default while debugging. Hiding labels makes it much harder to see
    # what the tracker is doing wrong, and a hidden marker and a missing marker
    # look identical on screen. Turn it back on for the deliverable.
    ap.add_argument("--ball-fade", action="store_true",
                    help="fade the numbers of players far from the ball")
    # Diagnostic, never a deliverable. Boxes are ~37% too short in every model
    # tested; this says WHICH edge is wrong, and the two have opposite fixes.
    ap.add_argument("--boxes", action="store_true",
                    help="draw the raw bounding box instead of the foot ellipse "
                         "— CYAN top edge, MAGENTA bottom edge")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH")
    if not args.tracks.exists():
        sys.exit(f"no such file: {args.tracks}")

    T = json.loads(args.tracks.read_text(encoding="utf-8"))
    clip = args.clip or Path("clips") / T["clip"]
    if not clip.exists():
        sys.exit(f"clip not found: {clip} (pass --clip)")

    W, H, fps = probe_size(clip)

    # team letter -> drawn colour
    kits = list(T.get("teams", {}).keys())
    kit_a = kits[0] if kits else None
    kit_b = kits[1] if len(kits) > 1 else None
    ca, cb, note = resolve_team_colours(kit_a, kit_b, T.get("accents", {}))
    colours = {T["teams"].get(kit_a): ca}
    if kit_b:
        colours[T["teams"].get(kit_b)] = cb

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.tracks.stem.replace('__tracks', '') + ('__BOXES' if args.boxes else '')
    out = args.out or OUT_DIR / f"{stem}.mp4"

    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(clip),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE)

    by_frame = {f["frame"]: f for f in T["frames"]}
    blank = {"players": [], "ball": None}
    font_cache, n, nbytes = {}, 0, W * H * 3
    try:
        while True:
            buf = reader.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            img = Image.frombytes("RGB", (W, H), buf)
            img = draw_frame(img, by_frame.get(n, blank), colours, W, H,
                             font_cache, args.ball_fade, args.boxes)
            writer.stdin.write(img.tobytes())
            n += 1
    finally:
        if writer.stdin:
            writer.stdin.close()
        writer.wait()
        reader.stdout.close()
        reader.wait()

    drawn = sum(len(f["players"]) for f in T["frames"])
    withball = sum(1 for f in T["frames"] if f["ball"])
    onball = sum(1 for f in T["frames"] for p in f["players"] if p.get("on_ball"))
    print(f"\n  teams           {T.get('teams')}")
    print(f"  accents         {T.get('accents') or '(none reported)'}")
    print(f"  colour rule     {note}")
    print(f"  team A colour   rgb{colours.get('A')}")
    print(f"  team B colour   rgb{colours.get('B')}")
    print(f"  frames written  {n} of {T['n_frames']} expected")
    print(f"  markers drawn   {drawn} player-frames, avg "
          f"{drawn / max(n,1):.1f} per frame")
    print(f"  ball drawn      {withball} frames")
    print(f"  on-ball drawn   {onball} player-frames")
    unknown = {k for k in T.get("teams", {}) if named(k) is None}
    if unknown:
        print(f"  UNKNOWN COLOURS {unknown} — rendered grey, add them to NAMED")
    print(f"\n  -> {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
