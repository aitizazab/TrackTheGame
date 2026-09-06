"""
Collect and normalise 30-second clips for the tracker.

Two jobs, and the second matters more than it looks:

  DOWNLOAD  pull candidate footage with yt-dlp, preferring 1080p — detection
            runs at 1080p (D19) and the raw file is gitignored, so there is
            nothing to save by pulling it smaller.

  NORMALISE cut exactly 30.0s, force CONSTANT 30fps, drop audio, and write TWO
            files: <name>.mp4 at 1280x720 for rendering and committing, and
            <name>_1080.mp4 at 1920x1080 for detection. Every clip then has
            exactly 900 frames and frame index N means the same instant in
            every clip.

Why normalising is not optional: source footage is routinely variable-frame-rate.
Extract frames from a VFR file by index and the timestamps drift, so the boxes a
model returns for "frame 300" get drawn onto a different moment than the one the
model saw. The annotation slides out of sync with the video and it looks like a
tracking bug. Force CFR once, up front, and the whole class of problem is gone.

    uv run fetch_clips.py list                      # show the candidate pool
    uv run fetch_clips.py get football_wide         # one candidate
    uv run fetch_clips.py get all                   # the whole pool
    uv run fetch_clips.py normalise raw/foo.mp4 --start 01:12 --name my_clip
    uv run fetch_clips.py verify                    # confirm 900 frames each

Curation is yours. Pull a wide pool, watch them, keep five that differ from each
other — and keep at least one you expect to be hard.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

RAW = Path("clips/raw")
OUT = Path("clips")
TARGET_W, TARGET_H, TARGET_FPS, TARGET_SECS = 1280, 720, 30, 30
EXPECTED_FRAMES = TARGET_FPS * TARGET_SECS  # 900

# Candidate pool. Deliberately spread across the axes that actually stress the
# task: sport, camera distance, lighting, and how similar the two kits are.
# `start` is where the useful 30 seconds begins in the source.
# NOTE: `download()` sorts with -S res:720, which is a PREFERENCE, not a cap. We
# ship detection at 1080p (D3/D19), so every clip below was pulled at 1080 and
# normalised twice: <name>.mp4 at 1280x720 for rendering, <name>_1080.mp4 at
# 1920x1080 for detection. Coordinates are fractions, so the two are independent.
CANDIDATES = {
    # name              url                                                     start    why it is in the pool
    "allstars_fr_eng": ("",                                                     "65:35", "SHIPPED, clip 1 of 5 (D19). England v France, broadcast wide. URL WAS NEVER RECORDED — ask the user"),
    "basketball":      ("https://www.youtube.com/watch?v=5U9k1U6nN-g",          "00:05", "indoor court, 10 players not 22, large legible numbers — the number layer at the opposite extreme"),
    "football_amateur": ("https://www.youtube.com/watch?v=CNhrwaChUAA",         "02:56", "amateur match, no broadcast grade, uneven exposure"),
    "football_cuts":   ("https://www.youtube.com/watch?v=OT3rAWUqOjU",          "00:00", "SHIPPED, clip 4 of 5. FIVE HARD CUTS: 17.2s, 18.3s, 21.6s and 29.4s are detected; the one at 8.8s is NOT and cannot be (D33). The only footage that exercises D8"),
    "volleyball":      ("https://www.youtube.com/watch?v=0lN1HfFAYUY",          "00:00", "SHIPPED, clip 5 of 5, user-chosen. A THIRD SPORT: no goalkeepers, a net splitting the two teams, and a ball that is airborne almost continuously - the hardest possible case for the possession rule. Passed: the kit vote found the liberos unaided (white 705, blue 700, red 135, green 104) and the referee on the stand was correctly never detected"),
    # ALL FIVE CLIPS COLLECTED, 5 Sep. The three below were never sourced and are
    # kept only as a record of what the pool was aiming at. '!' in `list` means
    # the URL is not filled in.
    #   - football_similar_kits: D11's dE>=30 rule STILL has never fired on real
    #     footage, so the similar-kit case remains untested. Volleyball was taken
    #     as clip 5 instead, on the grounds that a third SPORT tests more of the
    #     design than a fourth football clip would.
    #   - football_setpiece / football_pan_zoom: superseded. Crowding is covered
    #     by allstars, continuous hard pan by football_amateur.
    "football_similar_kits": ("https://www.youtube.com/watch?v=",               "00:25", "NOT COLLECTED. Kits close in colour - D11's dE>=30 fallback has never fired on any real footage"),
    "football_setpiece": ("https://www.youtube.com/watch?v=",                   "00:00", "NOT COLLECTED, superseded by allstars. Corner or free kick: 15+ players in the box, the association worst case"),
    "football_pan_zoom": ("https://www.youtube.com/watch?v=",                   "00:00", "NOT COLLECTED, superseded by football_amateur. A zoom changes every box height at once, attacking the depth cue and the gate together"),
}


def need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        sys.exit(f"{tool} not found on PATH. Restart your shell if you just installed it.")
    return path


def run(cmd: list, quiet: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False,
                          stdout=subprocess.PIPE if quiet else None,
                          stderr=subprocess.STDOUT if quiet else None, text=True)


def download(name: str, url: str) -> Path:
    need("yt-dlp")
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f"{name}.%(ext)s"
    print(f"  downloading {name} ...")
    # RAISED TO 1080p, 5 Sep. The cap said 720 because "we normalise to
    # 1280x720 anyway" - which stopped being true at D19, when detection moved
    # to 1080p and every clip started being normalised TWICE. The comment at the
    # top of this file already claimed 1080; the code had never been updated, so
    # the volleyball clip came down at 720 and could not produce a _1080 detect
    # input. Resolution is free in input tokens on this model but measurably
    # worth 7.9% -> 12.2% on jersey numbers, so the cap was costing accuracy.
    r = run(["yt-dlp", "-f", "bv*+ba/b", "-S", "res:1080,ext:mp4:m4a",
             "--merge-output-format", "mp4", "-o", str(target), url])
    if r.returncode != 0:
        print(f"  FAILED {name}: {(r.stdout or '').strip().splitlines()[-1:]}")
        return None
    hits = list(RAW.glob(f"{name}.*"))
    return hits[0] if hits else None


def normalise(src: Path, start: str, name: str,
              w: int = None, h: int = None, suffix: str = "") -> Path:
    """Cut exactly 30s at CFR 30fps, no audio, at the requested size.

    Called TWICE per clip: <name>.mp4 at 1280x720 is the render target, and
    <name>_1080.mp4 at 1920x1080 is the detection input. Coordinates are
    fractions, so the two are independent - detecting at 1080 while rendering
    at 720 keeps the deliverable ~14MB instead of ~40MB (D19).
    """
    need("ffmpeg")
    OUT.mkdir(parents=True, exist_ok=True)
    w = w or TARGET_W
    h = h or TARGET_H
    dst = OUT / f"{name}{suffix}.mp4"
    # -ss before -i seeks fast; because we re-encode, ffmpeg still lands
    # frame-accurate rather than snapping to the previous keyframe.
    # force_original_aspect_ratio + pad keeps geometry undistorted: a squashed
    # player is a player the model has never seen the shape of.
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={TARGET_FPS}")
    cmd = ["ffmpeg", "-y", "-ss", start, "-i", str(src), "-t", str(TARGET_SECS),
           "-vf", vf, "-fps_mode", "cfr", "-r", str(TARGET_FPS),
           "-c:v", "libx264", "-preset", "medium", "-crf", "23",
           "-pix_fmt", "yuv420p", "-an", str(dst)]
    r = run(cmd)
    if r.returncode != 0:
        print(f"  ffmpeg failed on {src.name}:\n{(r.stdout or '')[-600:]}")
        return None
    print(f"  wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def probe(path: Path) -> dict:
    need("ffprobe")
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries",
             "stream=nb_read_frames,r_frame_rate,avg_frame_rate,width,height",
             "-of", "json", str(path)])
    try:
        return json.loads(r.stdout)["streams"][0]
    except Exception:
        return {}


def cmd_list(_):
    print(f"\n  {'name':<24} {'start':<7} why\n  " + "-" * 78)
    for n, (url, start, why) in CANDIDATES.items():
        mark = " " if url.rstrip("=") != url.rstrip("=").rstrip("watch?v") else "!"
        print(f" {mark}{n:<24} {start:<7} {why}")
    print("\n  '!' = URL not filled in yet. Paste real URLs into CANDIDATES first.")


def cmd_get(args):
    names = list(CANDIDATES) if args.name == "all" else [args.name]
    for n in names:
        if n not in CANDIDATES:
            print(f"  unknown candidate: {n}")
            continue
        url, start, _ = CANDIDATES[n]
        if url.endswith("watch?v="):
            print(f"  skipping {n}: no URL set")
            continue
        src = download(n, url)
        if src:
            normalise(src, start, n)                       # 720p render target
            normalise(src, start, n, 1920, 1080, "_1080")  # 1080p detect input


def cmd_normalise(args):
    normalise(Path(args.src), args.start, args.name)


def cmd_verify(_):
    clips = sorted(p for p in OUT.glob("*.mp4"))
    if not clips:
        sys.exit("no clips found in clips/")
    print(f"\n  {'clip':<28} {'frames':>7} {'fps':>10} {'size':>10} ok")
    print("  " + "-" * 68)
    for c in clips:
        s = probe(c)
        frames = int(s.get("nb_read_frames", 0) or 0)
        ok = (frames == EXPECTED_FRAMES
              and int(s.get("width", 0)) == TARGET_W
              and int(s.get("height", 0)) == TARGET_H)
        print(f"  {c.name:<28} {frames:>7} {s.get('r_frame_rate','?'):>10} "
              f"{s.stat if False else f'{c.stat().st_size/1e6:.1f} MB':>10} "
              f"{'YES' if ok else 'NO'}")
    print(f"\n  Every clip must read exactly {EXPECTED_FRAMES} frames at "
          f"{TARGET_W}x{TARGET_H}. Anything else will desync the annotation.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    g = sub.add_parser("get"); g.add_argument("name"); g.set_defaults(fn=cmd_get)
    n = sub.add_parser("normalise")
    n.add_argument("src"); n.add_argument("--start", default="00:00")
    n.add_argument("--name", required=True); n.set_defaults(fn=cmd_normalise)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
