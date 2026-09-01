"""Does resolution change what Gemini charges — and what it can read?

The hypothesis, from the previous project: Gemini bills images by TILE, not by
pixel, so input tokens are quantised rather than continuous. If true, sending a
higher-resolution frame costs little or nothing extra while giving the model
more pixels on the thing it is worst at — an 8-pixel jersey number.

Two supporting observations already in hand:
  - gemini-3.7-flash and gemini-3.5-flash-lite both report EXACTLY 2821 input
    tokens for the same 1280x720 frame. Two different models landing on the same
    count is what tiling looks like; per-pixel billing would not do that.
  - Luna, by contrast, measured as continuous: tokens ~= 0.00119 x pixels + 19.
    So this is a per-vendor property and has to be measured per vendor.

The source footage is 1920x1080; fetch_clips.py has been throwing away 2.25x the
pixels. If tokens really are flat, that was free resolution we discarded.

Sweeps width on ONE clip so nothing else changes, and reports the two numbers
that decide it: input tokens, and the jersey-number read rate.

    uv run probe_resolution.py clips/hard10_allstars_1080.mp4 \
        --model google/gemini-3.7-flash --effort low
"""

import argparse
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

WIDTHS = [640, 960, 1280, 1920]
FRAMES = "0,36,72,108,144,180,216,252"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--widths", default=None)
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",")] if args.widths else WIDTHS
    short = re.sub(r"[^a-z0-9]+", "_", args.model.lower()).strip("_")[:22]
    n = len(args.frames.split(","))
    print(f"{args.model}  effort={args.effort or 'default'}")
    print(f"{len(widths)} widths x {n} frames = {len(widths)*n} calls\n")
    print(f"{'width':>6} {'in tok':>7} {'out tok':>8} {'players':>8} {'NUMS':>7} "
          f"{'ball':>6} {'lat':>6} {'$/frame':>8}")
    print("-"*66)

    base = None
    for w in widths:
        tag = f"res_{short}_{w}"
        cmd = [sys.executable, "detect.py", str(args.clip), "--model", args.model,
               "--frames", args.frames, "--tag", tag, "--width", str(w)]
        if args.effort:
            cmd += ["--effort", args.effort]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{w:>6}  FAILED: {(r.stdout or '')[-200:]}")
            continue

        rows = [json.loads(l) for l in Path("docs/run_log.jsonl")
                .open(encoding="utf-8") if l.strip()]
        mine = [x for x in rows if x.get("tag") == tag]
        ok = [x for x in mine if x.get("ok")]
        it = [x["prompt_tokens"] for x in ok if x.get("prompt_tokens")]
        ot = [x["completion_tokens"] for x in ok if x.get("completion_tokens")]
        lat = [x["latency_s"] for x in mine if x.get("latency_s")]
        cost = sum(x.get("cost_usd") or 0 for x in mine)

        det = Path("outputs/detections")/f"{args.clip.stem}__{tag}.json"
        nums = pl = ball = fr = 0
        if det.exists():
            d = json.loads(det.read_text(encoding="utf-8"))
            fr = len(d["frames"]) or 1
            ps = [q for f in d["frames"] for q in f["players"]]
            pl = len(ps)
            nums = sum(1 for q in ps if q.get("num") is not None)
            ball = sum(1 for f in d["frames"] if f.get("ball"))
        tok = st.median(it) if it else 0
        if base is None and tok:
            base = tok
        print(f"{w:>6} {tok:>7.0f} {st.median(ot) if ot else 0:>8.0f} "
              f"{pl/max(fr,1):>8.1f} {nums/max(pl,1)*100:>6.1f}% "
              f"{ball/max(fr,1)*100:>5.0f}% {st.median(lat) if lat else 0:>5.1f}s "
              f"{cost/max(len(mine),1):>8.5f}")

    print(f"\n  If 'in tok' is FLAT across widths, resolution is free and we")
    print(f"  should be sending 1920 — the source is 1080p and fetch_clips.py")
    print(f"  has been downscaling to 720p for no reason.")
    print(f"  If 'NUMS' climbs with width, that is the jersey-number fix.")


if __name__ == "__main__":
    main()
