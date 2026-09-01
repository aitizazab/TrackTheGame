"""Run several models over one clip and put the results side by side.

Everything the comparison needs is already recorded by detect.py and track.py —
this only sequences the runs and reads the numbers back, so a screen is one
command instead of three per model plus a spreadsheet.

Each model gets its own tag, so `docs/run_log.jsonl` and
`outputs/detections/<clip>__<tag>.json` stay separate and re-runnable.

    uv run screen.py clips/hard10_allstars.mp4
    uv run screen.py clips/hard10_allstars.mp4 --models a,b,c
    uv run screen.py clips/hard10_allstars.mp4 --dry-run     # cost estimate only
    uv run screen.py clips/hard10_allstars.mp4 --render      # also make videos

Costs real money. --dry-run first.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Tier A shortlist. Cheap enough that all three together cost less than one
# Luna pass on the 30s clip. See docs/decisions.md for why these three.
DEFAULT = [
    "qwen/qwen3-vl-30b-a3b-instruct",  # sparse-MoE sibling of the 32B: as good, faster?
    "google/gemini-3.7-flash",         # newest flash line, cheaper than 3.5 and 3.6
    "google/gemini-3.5-flash-lite",    # newest LITE line, dearer than 3.7-flash
    "google/gemini-3.1-flash-lite",    # re-run: last time it returned pixels
]

# From docs/vision_models.json, $/Mtok in / $/Mtok out / does it reason.
PRICES = {
    "qwen/qwen3-vl-32b-instruct":      (0.104, 0.416, False),
    "qwen/qwen3-vl-30b-a3b-instruct":  (0.130, 0.520, False),
    "qwen/qwen3.5-9b":                 (0.100, 0.150, True),
    # Flagged reasoning=False from OBSERVATION, not from the catalogue: both
    # reported zero reasoning tokens across a full run. The catalogue flag means
    # the model *can* reason, which is a different claim.
    "google/gemini-2.5-flash-lite":    (0.100, 0.400, False),
    "google/gemini-3.1-flash-lite":    (0.250, 1.500, False),
    "google/gemini-3.7-flash":         (0.375, 1.875, True),   # unobserved
    "google/gemini-3.5-flash-lite":    (0.300, 2.500, True),   # unobserved
    "openai/gpt-5.6-luna":             (0.200, 1.200, True),
}

# Corrected from the first screen. The catalogue's "reasoning" flag means
# SUPPORTED, not on-by-default: all three models that returned in the first run
# reported zero reasoning tokens and emitted 1456-1654 output tokens of pure
# content. The old 750-token "terse" figure was guesswork and 62% low; 2035 for
# reasoning models was Luna's behaviour, not everyone's.
IN_TOK, OUT_REASON, OUT_TERSE = 2188, 2256, 1550


def tag_for(model):
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")[:40]


def run(cmd):
    print(f"    $ {' '.join(cmd[-6:])}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FAILED:\n{(r.stdout or '')[-500:]}\n{(r.stderr or '')[-500:]}")
        return None
    return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--models", default=None, help="comma-separated overrides")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=43.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    models = args.models.split(",") if args.models else DEFAULT
    if not args.clip.exists():
        sys.exit(f"no such clip: {args.clip}")

    # crude frame count: duration x fps
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(args.clip)], capture_output=True, text=True)
    secs = float(probe.stdout.strip() or 30)
    frames = int(round(secs * args.fps))

    print(f"clip {args.clip.name}  {secs:.1f}s  -> {frames} frames at {args.fps}fps\n")

    # Surface unpinned models HERE, not 40 calls into a paid run. detect.py
    # refuses them by design; there is no point discovering that halfway through.
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from detect import COORD_CONVENTION
        unpinned = [m for m in models if m not in COORD_CONVENTION]
    except Exception:
        unpinned = []
    if unpinned:
        print("  UNPINNED — these have no coordinate convention and will not run:")
        for m in unpinned:
            print(f"    uv run detect.py {args.clip} --model {m} --probe-convention")
        print("  Three calls each. Add the printed line to COORD_CONVENTION, "
              "then re-run this.\n")

    total = 0.0
    for m in models:
        p = PRICES.get(m)
        if not p:
            print(f"  {m:<34}  price unknown — check docs/vision_models.json")
            continue
        out_tok = OUT_REASON if p[2] else OUT_TERSE
        est = frames * IN_TOK / 1e6 * p[0] + frames * out_tok / 1e6 * p[1]
        total += est
        print(f"  {m:<34}  est ${est:.3f}"
              f"   ({'reasoning' if p[2] else 'terse'} output assumed)")
    print(f"  {'TOTAL':<34}  est ${total:.3f}\n")
    if args.dry_run:
        print("dry run — nothing sent.")
        return

    results = []
    for m in models:
        tag = tag_for(m)
        print(f"\n=== {m} ===")
        t0 = time.perf_counter()
        out = run([sys.executable, "detect.py", str(args.clip), "--fps",
                   str(args.fps), "--model", m, "--tag", tag,
                   "--timeout", str(args.timeout)])
        if out is None:
            results.append({"model": m, "tag": tag, "failed": True})
            continue
        print(out.rstrip())
        djson = Path("outputs/detections") / f"{args.clip.stem}__{tag}.json"
        tout = run([sys.executable, "track.py", str(djson)])
        if tout:
            print(tout.rstrip())
        if args.render:
            tj = Path("outputs/tracks") / f"{djson.stem}__tracks.json"
            run([sys.executable, "render.py", str(tj)])
        results.append({"model": m, "tag": tag, "wall": time.perf_counter() - t0,
                        "det": djson})

    # ---- side by side -----------------------------------------------------
    print(f"\n\n{'model':<34} {'wall':>7} {'ret':>6} {'cost':>8} {'p/fr':>6} "
          f"{'nums':>7} {'ball':>6} {'lat p50':>8}")
    print("-" * 92)
    for r in results:
        if r.get("failed"):
            print(f"{r['model'][:34]:<34}  FAILED")
            continue
        d = json.loads(Path(r["det"]).read_text(encoding="utf-8"))
        log = [json.loads(l) for l in Path("docs/run_log.jsonl")
               .open(encoding="utf-8") if l.strip()]
        mine = [x for x in log if x.get("tag") == r["tag"]]
        cost = sum(x.get("cost_usd") or 0 for x in mine)
        lat = sorted(x["latency_s"] for x in mine if x.get("latency_s"))
        sight = sum(len(f["players"]) for f in d["frames"])
        nums = sum(1 for f in d["frames"] for p in f["players"]
                   if p.get("num") is not None)
        ball = sum(1 for f in d["frames"] if f.get("ball"))
        n = max(len(d["frames"]), 1)
        print(f"{r['model'][:34]:<34} {r['wall']:>7.1f} "
              f"{len(d['frames']):>3}/{len(d['frames'])+len(d['dropped']):<2} "
              f"${cost:>7.3f} {sight/n:>6.1f} {nums/max(sight,1)*100:>6.1f}% "
              f"{ball/n*100:>5.0f}% {lat[len(lat)//2] if lat else 0:>8.1f}")
    print("\nnums = share of player sightings with a legible jersey number — the "
          "number that decides\nwhether labels are real or invented.")


if __name__ == "__main__":
    main()
