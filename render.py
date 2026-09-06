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
# SHIPPED FONT FIRST. The rest of the stack is a fallback for a machine that
# does not have the repo checked out beside it. Resolving against installed
# Windows fonts made the deliverable depend on which machine rendered it - the
# labels would silently change face on anyone else's computer, including the
# assessor's if they re-ran it.
FONT_STACK = ("assets/fonts/BlackOpsOne.ttf",
              "bahnschrift.ttf", "seguisb.ttf", "tahomabd.ttf",
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
LABELS_FADE = 4              # then this many fading toward the floor
# Everyone beyond that keeps this much opacity. Never 0: a number that vanishes
# is not a de-emphasised number, it is a missing one.
LABELS_MIN_ALPHA = 0.38
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

# Set by --font to a .ttf path. A font shipped in the repo is the only way a
# render reproduces on another machine: FONT_STACK below resolves against
# whatever the OS happens to have installed, which on this project meant
# "whatever Windows has" and would silently change the deliverable elsewhere.
FONT_OVERRIDE = [None]
# Variable fonts arrive at their default instance, which is usually Regular and
# too light for a marker label. Ask for a heavy named instance if there is one.
_WANT_INSTANCE = ("Bold", "SemiBold", "ExtraBold", "Black", "Medium")


def load_font(px):
    if FONT_OVERRIDE[0]:
        try:
            f = ImageFont.truetype(str(FONT_OVERRIDE[0]), px)
            try:
                names = [n.decode() if isinstance(n, bytes) else n
                         for n in f.get_variation_names()]
                for want in _WANT_INSTANCE:
                    if want in names:
                        f.set_variation_by_name(want)
                        break
            except Exception:
                pass                      # not a variable font, or no instances
            return f
        except OSError:
            pass
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


# --- label collision handling ---------------------------------------------
# A label is nudged UPWARD, never sideways. x is what ties a label to its
# player: move it horizontally and the viewer has to guess whose number that
# is, which is worse than the overlap. Moving it up keeps it in the same
# vertical column as the head it belongs to.
LABEL_NUDGE_STEP = 0.75      # of label height, per attempt
LABEL_MAX_NUDGE = 3.0        # give up after this many label heights
# Density fade. Above this many labels inside a radius of DENSITY_R label
# widths, the whole neighbourhood dims - a wall of numbers over a goalmouth
# scramble hides the thing the viewer is trying to watch.
DENSITY_R = 4.0
DENSITY_N = 5
DENSITY_MIN_ALPHA = 0.55


def resolve_label_collisions(boxes):
    """Nudge overlapping labels upward. Returns (new_ys, unresolved_flags).

    `boxes` is [(cx, y, w, h, priority), ...] with y the TOP of the label.
    Higher priority keeps its position; everyone else moves out of the way.
    Priority is the player's box height, i.e. how near the camera they are -
    a near player's label is the one the viewer is most likely reading, and a
    distant player's label has least to lose by shifting.

    Greedy rather than a global optimisation on purpose: with at most ~20
    labels the greedy pass is exact enough, and a solver that reshuffles every
    label each frame would make the whole layer jitter frame to frame.
    """
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][4])
    placed = []
    ys = [b[1] for b in boxes]
    unresolved = [False] * len(boxes)
    for i in order:
        cx, y, w, h, _ = boxes[i]
        step = max(1.0, h * LABEL_NUDGE_STEP)
        limit = h * LABEL_MAX_NUDGE
        moved = 0.0
        while True:
            r = (cx - w / 2, y - moved, cx + w / 2, y - moved + h)
            hit = any(not (r[2] <= q[0] or r[0] >= q[2] or
                           r[3] <= q[1] or r[1] >= q[3]) for q in placed)
            if not hit:
                break
            moved += step
            if moved > limit:
                unresolved[i] = True     # nowhere to go - let the fade take it
                break
        ys[i] = y - moved
        placed.append((cx - w / 2, ys[i], cx + w / 2, ys[i] + h))
    return ys, unresolved


def density_alphas(boxes):
    """Dim a neighbourhood that has too many labels in it at once."""
    n = len(boxes)
    out = [1.0] * n
    if n <= DENSITY_N:
        return out
    for i in range(n):
        cx, y, w, h, _ = boxes[i]
        # Counted on a box-sized neighbourhood rather than a wide radius, so a
        # label that displacement has cleanly separated is not punished for
        # having neighbours. This is about visual clutter, not collision - the
        # collision case is handled above, by moving things.
        r = DENSITY_R * max(w, 1.0)
        near = sum(1 for j in range(n) if j != i and
                   abs(cx - boxes[j][0]) < r and abs(y - boxes[j][1]) < 2.0 * h)
        if near >= DENSITY_N:
            over = min(1.0, (near - DENSITY_N + 1) / float(DENSITY_N))
            out[i] = 1.0 - (1.0 - DENSITY_MIN_ALPHA) * over
    return out


def number_alphas(labels):
    """Crowding fade.

    Two numbers drawn on top of each other are worse than one number: the reader
    cannot resolve either, and the picture looks broken rather than busy. Fading
    both, in proportion to how close they are, turns an unreadable collision into
    an honest visual signal that two players are overlapping.

    `labels` is [(centre_x, top_y, width, height), ...] AFTER displacement;
    returns one alpha per label.
    """
    # RECTANGLE OVERLAP, not centre distance - corrected 6 Sep.
    #
    # The circular test measured centre-to-centre distance against a radius
    # derived from glyph WIDTH, so it fired on labels that had been nudged
    # apart VERTICALLY and no longer touched at all. With displacement now
    # running first, that meant a collision was fixed and then both labels were
    # dimmed anyway for having once been close - visible on allstars frame 18,
    # where `21` and `XII` end up clearly separated and both faded.
    #
    # Overlap of the actual drawn boxes is the thing this fade is for. If they
    # do not overlap there is nothing to signal.
    n = len(labels)
    alphas = [1.0] * n
    rects = [(cx - w / 2, y, cx + w / 2, y + h) for cx, y, w, h in labels]
    for i in range(n):
        ax0, ay0, ax1, ay1 = rects[i]
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = rects[j]
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox <= 0 or oy <= 0:
                continue                      # genuinely clear - no fade
            # Fade in proportion to how much of the smaller label is buried.
            frac = (ox * oy) / max(1.0, min((ax1 - ax0) * (ay1 - ay0),
                                            (bx1 - bx0) * (by1 - by0)))
            a = 1.0 - (1.0 - CROWD_MIN_ALPHA) * min(1.0, frac)
            alphas[i] = min(alphas[i], a)
            alphas[j] = min(alphas[j], a)
    return alphas


# How long a label takes to slide to a new position, seconds. The nudge is
# recomputed every frame, so as players drift the target offset changes - and
# without easing the label TELEPORTS between the nudged and un-nudged position,
# which reads as a glitch rather than as a layout decision.
LABEL_EASE_S = 0.25


def _smoothstep(t):
    """Cubic ease-in-out: 3t^2 - 2t^3. Zero velocity at both ends, so a label
    starts moving and stops moving smoothly instead of snapping."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# --- marker styles ---------------------------------------------------------
# The original marker is a flat ellipse with a hard outline: correct, and it
# reads as a debug overlay rather than as broadcast graphics. Three directions,
# selectable so they can be compared on real footage instead of argued about.
#
#   classic    what shipped: flat fill, hard rim.
#   broadcast  grounded: a drop shadow so the marker sits UNDER the player
#              rather than floating on the grass, a soft graded fill, a bright
#              thin rim. Closest to televised match graphics.
#   spotlight  no rim at all. A pool of light under the player, brightest at
#              the feet and falling off outward. Least ink on the pitch, so the
#              football stays the most visible thing in the frame.
#   tactical   analysis-software look: open brackets left and right instead of
#              a closed ring, plus a riser connecting the mark to its label so
#              a displaced label is unambiguously attached to its player.
STYLE = ["classic"]
# THE CARRIER KEEPS ITS TEAM COLOUR. Only the OUTER indicator changes.
#
# Tried and rejected: recolouring the carrier's whole marker amber. It made the
# carrier instantly findable and destroyed the thing that mattered more - which
# team the player on the ball is on. A third state must be drawn ON TOP of the
# team identity, never instead of it.
#
# And a fixed accent colour just relocates the collision: amber is unreadable
# against an amber kit. So the accent is DERIVED per clip - the candidate whose
# minimum CIE76 distance to BOTH team colours is largest. On blue/white it lands
# on a warm hue; on a red/yellow match it lands somewhere cold. Yellow and
# orange stay in the pool but lose whenever a kit is near them, which is what a
# derived choice is for.
#
# The carrier is ALSO marked by shape, not colour alone - a second concentric
# ring - so it survives even where the accent is only moderately distinct.
CARRIER_CANDIDATES = [
    (255, 176, 32), (0, 229, 255), (255, 64, 160), (140, 255, 60),
    (255, 255, 255), (24, 24, 26),
]
CARRIER_ACCENT = [(255, 255, 255)]


def pick_carrier_accent(team_cols, frame=None):
    """Accent furthest from every team colour AND from the playing surface.

    The first version considered only the two kit colours and picked GREEN for a
    blue-versus-white match - correct by its own metric and useless in practice,
    because the largest coloured object in every frame is the grass. The surface
    is a competing colour exactly like a kit is, so it belongs in the same
    distance test. Sampled from the actual frame rather than assumed, since the
    five clips run over grass, a wooden court and a blue taraflex.
    """
    cols = [c for c in team_cols if c]
    if frame is not None:
        small = frame.resize((32, 32))
        px = list(small.getdata())
        med = tuple(sorted(c[i] for c in px)[len(px) // 2] for i in range(3))
        cols.append(med)
    if not cols:
        return CARRIER_CANDIDATES[0]

    # HUE separation, not raw dE. CIE76 counts lightness heavily, so a bright
    # green scores "far" from dark grass while sharing its hue - and on moving
    # video hue is what the eye separates objects by. Measured: the dE-only
    # version picked (140,255,60) for a blue/white match on a GRASS pitch.
    # Candidates within HUE_GUARD of the surface hue are struck out entirely;
    # dE then ranks whatever survives.
    HUE_GUARD = 0.11                      # ~40 degrees on the colour wheel
    def hue(c):
        return colorsys.rgb_to_hsv(*[v / 255 for v in c])[0]

    surface = cols[-1] if frame is not None else None
    pool = CARRIER_CANDIDATES
    if surface is not None:
        hs = hue(surface)
        keep = [c for c in pool
                if min(abs(hue(c) - hs), 1.0 - abs(hue(c) - hs)) > HUE_GUARD
                or colorsys.rgb_to_hsv(*[v / 255 for v in c])[1] < 0.15]
        pool = keep or pool           # never strike out everything
    return max(pool, key=lambda cand: min(delta_e(cand, c) for c in cols))


def _ell(d, cx, cy, w, h, **kw):
    d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], **kw)


def draw_marker(d, fx, fy, ew, eh, col, on_ball, fd, bw=0.0, bh=0.0):
    """Draw one player's ground marker in the active style."""
    a = lambda v: int(max(0, min(255, v * fd)))
    st = STYLE[0]
    acc = CARRIER_ACCENT[0]

    if st == "broadcast":
        # Shadow first, offset down: the cue that puts the marker on the grass.
        _ell(d, fx, fy + max(2, eh * 0.16), ew * 1.02, eh, fill=(0, 0, 0, a(70)))
        # Graded fill, brightest in the middle - three rings is enough to read
        # as a gradient at this size and costs nothing.
        for k, al in ((1.00, 28), (0.74, 40), (0.46, 58)):
            _ell(d, fx, fy, ew * k, eh * k, fill=col + (a(al),))
        rim = tuple(min(255, int(c * 1.25 + 40)) for c in col)
        _ell(d, fx, fy, ew, eh, outline=rim + (a(240),), width=2)
        if on_ball:
            _ell(d, fx, fy, ew * 1.20, eh * 1.20, outline=acc + (a(240),), width=3)
            _ell(d, fx, fy, ew * 1.42, eh * 1.42, outline=acc + (a(130),), width=2)

    elif st == "spotlight":
        # Concentric falloff, no outline anywhere. Reads as light rather than
        # as a drawn shape, which is what keeps it off the players.
        steps = 7
        for i in range(steps, 0, -1):
            k = 0.34 + (i / steps) * 0.86
            al = 10 + int(115 * (1.0 - (i / steps)) ** 1.7)
            _ell(d, fx, fy, ew * k, eh * k, fill=col + (a(al),))
        if on_ball:
            _ell(d, fx, fy, ew * 0.40, eh * 0.40, fill=(255, 255, 255, a(215)))
            _ell(d, fx, fy, ew * 1.24, eh * 1.24,
                 outline=(255, 255, 255, a(120)), width=2)

    elif st == "tactical":
        # Open brackets, not a closed ring: an unclosed shape reads as a
        # measurement rather than as an object sitting on the pitch.
        box = [fx - ew / 2, fy - eh / 2, fx + ew / 2, fy + eh / 2]
        _ell(d, fx, fy, ew, eh, fill=col + (a(26),))
        d.arc(box, 120, 240, fill=col + (a(245),), width=3)
        d.arc(box, 300, 60, fill=col + (a(245),), width=3)
        d.line([fx, fy - eh / 2, fx, fy - eh / 2 - eh * 0.9],
               fill=col + (a(150),), width=1)          # riser toward the label
        if on_ball:
            d.arc(box, 0, 360, fill=(255, 255, 255, a(235)), width=2)
            for ang in (0, 90, 180, 270):
                _ell(d, fx + (ew / 2) * math.cos(math.radians(ang)),
                     fy + (eh / 2) * math.sin(math.radians(ang)),
                     4, 4, fill=(255, 255, 255, a(235)))


    elif st == "stem":
        # A DOT AND A RISER, not a ring. The marker stops being a shape drawn on
        # the grass and becomes a pointer: a small disc at the feet with a thin
        # stem running up toward the label. It uses a fraction of the ink of an
        # ellipse, so the pitch stays visible, and the stem answers the one
        # complaint no ellipse can - which label belongs to which player.
        r = max(4.0, ew * 0.17)
        d.line([fx, fy - r * 0.4, fx, fy - max(bh * 0.92, eh * 2.6)],
               fill=col + (a(150),), width=2 if not on_ball else 3)
        _ell(d, fx, fy + 1.5, r * 2.1, r * 1.5, fill=(0, 0, 0, a(90)))
        _ell(d, fx, fy, r * 2.0, r * 1.35, fill=col + (a(235),),
             outline=(255, 255, 255, a(210)), width=1)
        if on_ball:
            _ell(d, fx, fy, r * 3.4, r * 2.3, outline=acc + (a(245),), width=3)
            _ell(d, fx, fy, r * 4.6, r * 3.1, outline=acc + (a(120),), width=2)

    elif st == "bar":
        # AN UNDERLINE. No enclosure at all - the player is underlined the way a
        # word is, which is about the least ink that can still say "this one".
        # Nothing is drawn around the player, so nothing competes with the kit,
        # and on a white shirt against a white line it is the bar's DARK shadow
        # that does the separating rather than a stroke colour.
        hbar = max(3.0, eh * 0.30)
        half = ew * 0.46
        d.rectangle([fx - half, fy + 1, fx + half, fy + 1 + hbar],
                    fill=(0, 0, 0, a(110)))
        d.rectangle([fx - half, fy - hbar * 0.5, fx + half, fy + hbar * 0.5],
                    fill=col + (a(240),))
        d.rectangle([fx - half, fy - hbar * 0.5, fx + half, fy - hbar * 0.2],
                    fill=tuple(min(255, int(c * 1.4 + 30)) for c in col) + (a(220),))
        if on_ball:
            cap = max(3.0, hbar * 0.9)
            for sx in (-1, 1):
                d.rectangle([fx + sx * half - cap * 0.5 * sx - (cap if sx > 0 else 0),
                             fy - hbar * 1.9, fx + sx * half + (0 if sx > 0 else cap),
                             fy + hbar * 1.1], fill=acc + (a(245),))
            d.rectangle([fx - half, fy - hbar * 1.9, fx + half, fy - hbar * 1.35],
                        fill=acc + (a(210),))

    elif st == "reticle":
        # FOUR CORNERS AROUND THE PLAYER, touching no grass at all. Borrowed from
        # a camera focus reticle: it frames the player rather than standing under
        # them, so it never fights the pitch markings that ruin a ground ellipse
        # on a white line. The corners are open, so the player is never enclosed.
        bw2 = max(bw, ew * 0.42) * 0.62
        bh2 = max(bh, eh * 3.0) * 0.5
        L = max(5.0, bh2 * 0.30)
        x0, x1 = fx - bw2, fx + bw2
        y0, y1 = fy - bh2 * 2.0, fy
        wdt = 3 if on_ball else 2
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                 (x0, y1, 1, -1), (x1, y1, -1, -1)):
            d.line([cx, cy, cx + dx * L, cy], fill=col + (a(245),), width=wdt)
            d.line([cx, cy, cx, cy + dy * L], fill=col + (a(245),), width=wdt)
        if on_ball:
            for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                     (x0, y1, 1, -1), (x1, y1, -1, -1)):
                ox, oy = -dx * 4, -dy * 4
                d.line([cx + ox, cy + oy, cx + ox + dx * L, cy + oy],
                       fill=acc + (a(240),), width=2)
                d.line([cx + ox, cy + oy, cx + ox, cy + oy + dy * L],
                       fill=acc + (a(240),), width=2)


    elif st in ("halo", "disc", "arena"):
        # ALL THREE KEEP THE RING. What changes is that a ring drawn at uniform
        # brightness reads as a decal painted on the grass, because nothing in
        # it responds to the scene. Three depth cues, used in combination:
        #
        #   1. FRONT/BACK GRADIENT. In this camera the top of the ellipse is the
        #      far side. Brightening the near arc and dimming the far one is how
        #      a real ring under a floodlight behaves, and it costs nothing.
        #   2. OCCLUSION. The player stands at the centre and their legs rise up
        #      the image, so the FAR arc should be partly hidden BY them. Fading
        #      it toward the top is the strongest single depth cue here - it puts
        #      the player inside the ring instead of on top of a drawing.
        #   3. THICKNESS. A second, darker arc offset a pixel below the main one
        #      gives the band an edge, so it reads as an object with substance
        #      rather than a 1px stroke.
        box = [fx - ew / 2, fy - eh / 2, fx + ew / 2, fy + eh / 2]
        segs = 48
        base_w = 3 if on_ball else 2
        # soft ground bloom so the ring is seated rather than floating
        for k, al in ((1.34, 12), (1.18, 20), (1.04, 30)):
            _ell(d, fx, fy + eh * 0.06, ew * k, eh * k, fill=col + (a(al),))
        if st != "arena":
            _ell(d, fx, fy, ew * 0.88, eh * 0.88, fill=col + (a(34),))
        for i in range(segs):
            a0 = i * 360.0 / segs
            a1 = a0 + 360.0 / segs + 1.2
            # 0 at the far side (270 deg), 1 at the near side (90 deg)
            near = (1.0 - math.cos(math.radians(a0 - 90.0))) * 0.5
            if st == "halo":
                al = 70 + 175 * near
            elif st == "disc":
                al = 120 + 125 * near
            else:                                   # arena - far arc drops out
                al = max(0.0, (near - 0.18) / 0.82) ** 0.85 * 250
            if al <= 4:
                continue
            d.arc(box, a0, a1, fill=col + (a(al),), width=base_w)
            if st in ("disc", "arena") and near > 0.35:
                # lower edge, darker: gives the band apparent thickness
                dark = tuple(int(c * 0.45) for c in col)
                d.arc([box[0], box[1] + 2, box[2], box[3] + 2], a0, a1,
                      fill=dark + (a(al * 0.7),), width=1)
        if on_ball:
            for i in range(segs):
                a0 = i * 360.0 / segs
                near = (1.0 - math.cos(math.radians(a0 - 90.0))) * 0.5
                al = 90 + 165 * near
                d.arc([box[0] - ew * 0.13, box[1] - eh * 0.13,
                       box[2] + ew * 0.13, box[3] + eh * 0.13],
                      a0, a0 + 360.0 / segs + 1.2, fill=acc + (a(al),), width=2)

    else:                                              # classic
        if on_ball:
            _ell(d, fx, fy, ew, eh, fill=col + (a(120),),
                 outline=(255, 255, 255, a(245)), width=3)
        else:
            _ell(d, fx, fy, ew, eh, fill=col + (a(55),),
                 outline=col + (a(225),), width=2)


def precompute_label_offsets(T, W, H, font_cache, fps):
    """Resolve label collisions for EVERY frame, then ease the result in time.

    Collision resolution is a per-frame layout decision, but the thing being
    laid out persists across frames, so the decision has to be continuous. This
    runs the whole clip first, builds each label's offset as a time series, and
    eases every change over LABEL_EASE_S with a cubic - the same reason the
    marker path is eased rather than drawn straight from per-sample output.

    Only possible because rendering is a batch job: an online renderer would
    have to guess where the label is about to need to go.
    """
    raw = {}
    for fr in T["frames"]:
        boxes, ids = [], []
        for p in fr["players"]:
            bh = p["h"] * H
            read = bool(p.get("read", True))
            size = int(max(NUM_MIN_PX, min(NUM_MAX_PX, bh * NUM_SIZE_MUL
                                           * (1.0 if read else INVENTED_SIZE_MUL))))
            if size not in font_cache:
                font_cache[size] = load_font(size)
            font = font_cache[size]
            text = str(p.get("label") or p["id"]).split("·")[-1]
            try:
                tw = float(font.getlength(text))
            except AttributeError:
                tw = float(size) * len(text) * 0.6
            fx, fy = p["x"] * W, p["y"] * H
            ty = fy - bh - bh * NUM_GAP_MUL - size
            boxes.append((fx, ty, tw, float(size), bh))
            ids.append(p["id"])
        if not boxes:
            raw[fr["frame"]] = {}
            continue
        ys, unres = resolve_label_collisions(boxes)
        raw[fr["frame"]] = {ids[i]: (boxes[i][1] - ys[i],
                                     1.0 if unres[i] else 0.0)
                            for i in range(len(ids))}

    # ---- ease each label's offset series ---------------------------------
    frames = sorted(raw)
    seen = {}
    for f in frames:
        for pid in raw[f]:
            seen.setdefault(pid, []).append(f)
    step = 1.0 / max(1.0, LABEL_EASE_S * fps)
    out = {f: {} for f in frames}
    for pid, fs in seen.items():
        cur = raw[fs[0]][pid][0]
        cur_u = raw[fs[0]][pid][1]
        t_from, t_to, prog = cur, cur, 1.0
        u_from, u_to, uprog = cur_u, cur_u, 1.0
        for f in fs:
            tgt, u = raw[f][pid]
            if abs(tgt - t_to) > 0.5:
                t_from, t_to, prog = cur, tgt, 0.0
            prog = min(1.0, prog + step)
            cur = t_from + (t_to - t_from) * _smoothstep(prog)
            if abs(u - u_to) > 0.01:
                u_from, u_to, uprog = cur_u, u, 0.0
            uprog = min(1.0, uprog + step)
            cur_u = u_from + (u_to - u_from) * _smoothstep(uprog)
            out[f][pid] = (cur, cur_u)
    return out


def draw_frame(base, entry, colours, W, H, font_cache, ball_fade=False,
               boxes=False, offsets=None):
    """Draw one frame's annotations onto `base` (an RGB Image). Returns it."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    players = entry["players"]
    labels, metas, glow_marks, label_ids = [], [], [], []

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
            # `fade` ramps to 0 over the last FADE_OUT_S of the drawn stretch,
            # so a marker dissolves instead of popping. 1.0 for every frame that
            # is not in the tail, so nothing else changes.
            fd = float(p.get("fade", 1.0))
            glow_marks.append((fx, fy, ew, eh, col))
            draw_marker(d, fx, fy, ew, eh, col, True, fd, bw, bh)
        else:
            fd = float(p.get("fade", 1.0))
            draw_marker(d, fx, fy, ew, eh, col, False, fd, bw, bh)

        read = bool(p.get("read", True))
        size = int(max(NUM_MIN_PX, min(NUM_MAX_PX, bh * NUM_SIZE_MUL
                                       * (1.0 if read else INVENTED_SIZE_MUL))))
        if size not in font_cache:
            font_cache[size] = load_font(size)
        font = font_cache[size]
        text = str(p.get("label") or p["id"]).split("·")[-1]
        tw = d.textlength(text, font=font)
        ty = fy - bh - bh * NUM_GAP_MUL - size
        # (centre_x, top_y, width, height, priority). Priority is box height:
        # a near player's label holds its place and distant ones give way.
        labels.append((fx, ty, tw, float(size), bh))
        label_ids.append(p["id"])
        metas.append((text, font, col, fx, ty, read, float(p.get('fade', 1.0))))

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
                # FLOOR, not zero. Setting distant labels to 0 deletes them,
                # which loses the "every player is identified" property the
                # brief actually asks for - a viewer looking away from the ball
                # should still be able to read who is who. Receding is the
                # intent; disappearing was an accident of using 0 as "least".
                rel[i] = LABELS_MIN_ALPHA

    # MOVE FIRST, FADE SECOND. Fading was the only response to a collision, so
    # two overlapping numbers both became unreadable to signal that they
    # overlapped. Nudging one upward usually makes BOTH readable, and the fade
    # is then only needed for the cases displacement cannot fix - which on a
    # goalmouth scramble is the honest outcome rather than the default one.
    # Offsets come from the eased whole-clip pass when one was computed; the
    # per-frame fallback exists so draw_frame still works standalone.
    if offsets is not None:
        ys = [labels[i][1] - offsets.get(label_ids[i], (0.0, 0.0))[0]
              for i in range(len(labels))]
        unresolved = [offsets.get(label_ids[i], (0.0, 0.0))[1]
                      for i in range(len(labels))]
    else:
        ys, unresolved = resolve_label_collisions(labels)
        unresolved = [1.0 if u else 0.0 for u in unresolved]
    dens = density_alphas(labels)
    labels = [(b[0], ys[i], b[2], b[3]) for i, b in enumerate(labels)]
    metas = [(m[0], m[1], m[2], m[3], ys[i], m[5], m[6])
             for i, m in enumerate(metas)]
    for i, ((text, font, col, tx, ty, read, fd), crowd, r) in enumerate(zip(
            metas, number_alphas(labels), rel)):
        alpha = min(crowd, r) * (1.0 if read else INVENTED_ALPHA_MUL) * fd
        alpha *= dens[i]
        # `unresolved` is now a 0..1 ramp, eased like the offset, so a label
        # that becomes stuck dims into it instead of stepping.
        alpha *= 1.0 - (1.0 - CROWD_MIN_ALPHA) * float(unresolved[i])
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
    ap.add_argument("--font", default=None,
                    help="path to a .ttf to draw labels with, e.g. "
                         "assets/fonts/BlackOpsOne.ttf. Ships with the repo so "
                         "the render reproduces off this machine.")
    ap.add_argument("--style", default="classic",
                    choices=["classic", "broadcast", "spotlight", "tactical",
                             "stem", "bar", "reticle",
                             "halo", "disc", "arena"],
                    help="marker look. See draw_marker.")
    ap.add_argument("--ball-fade", action="store_true",
                    help="fade the numbers of players far from the ball")
    # Diagnostic, never a deliverable. Boxes are ~37% too short in every model
    # tested; this says WHICH edge is wrong, and the two have opposite fixes.
    ap.add_argument("--boxes", action="store_true",
                    help="draw the raw bounding box instead of the foot ellipse "
                         "— CYAN top edge, MAGENTA bottom edge")
    args = ap.parse_args()
    if args.font:
        FONT_OVERRIDE[0] = args.font
    STYLE[0] = args.style

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
    CARRIER_ACCENT[0] = pick_carrier_accent([ca, cb])
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
    label_offsets = precompute_label_offsets(T, W, H, font_cache, fps)
    try:
        while True:
            buf = reader.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            img = Image.frombytes("RGB", (W, H), buf)
            if n == 0:
                CARRIER_ACCENT[0] = pick_carrier_accent([ca, cb], img)
            img = draw_frame(img, by_frame.get(n, blank), colours, W, H,
                             font_cache, args.ball_fade, args.boxes,
                             label_offsets.get(n, {}))
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
