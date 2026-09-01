"""Does the reasoning-effort control do anything on this model?

A5 is the largest cost lever available — reasoning is 42% of gemini-3.7-flash's
output tokens and 62% of Luna's, which is 36-53% of total spend. But the whole
ablation is worthless if the provider ignores the knob, and we have never
verified that it doesn't.

The one previous attempt was inconclusive rather than negative: it ran at
max_tokens=1, which had already truncated reasoning to 16 tokens, so there was
nothing for the flag to reduce. This runs at the real cap on real frames.

Five frames per level. The question is not "is it faster" — that is noisy at
n=5 — but "does reasoning_tokens MOVE". A provider that ignores the parameter
returns the same token count whatever you ask for, and that is unmistakable.

    uv run probe_effort.py clips/hard10_allstars.mp4 --model google/gemini-3.7-flash
    uv run probe_effort.py clips/hard10_allstars.mp4 --model openai/gpt-5.6-luna
"""

import argparse
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

LEVELS = [None, "low", "medium", "high"]   # None = no parameter sent at all
FRAMES = "0,60,120,180,240"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", default=FRAMES)
    args = ap.parse_args()

    short = re.sub(r"[^a-z0-9]+", "_", args.model.lower()).strip("_")[:24]
    n = len(args.frames.split(","))
    print(f"{args.model}: {len(LEVELS)} levels x {n} frames = "
          f"{len(LEVELS)*n} calls\n")

    for lvl in LEVELS:
        tag = f"eff_{short}_{lvl or 'none'}"
        cmd = [sys.executable, "detect.py", str(args.clip), "--model", args.model,
               "--frames", args.frames, "--tag", tag]
        if lvl:
            cmd += ["--effort", lvl]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {str(lvl or 'none'):<7} FAILED: {(r.stdout or '')[-300:]}")
            continue

        rows = [json.loads(l) for l in Path("docs/run_log.jsonl")
                .open(encoding="utf-8") if l.strip()]
        mine = [x for x in rows if x.get("tag") == tag]
        ok = [x for x in mine if x.get("ok")]
        rt = [x["reasoning_tokens"] for x in ok if x.get("reasoning_tokens") is not None]
        ct = [x["completion_tokens"] for x in ok if x.get("completion_tokens")]
        lat = [x["latency_s"] for x in mine if x.get("latency_s")]
        cost = sum(x.get("cost_usd") or 0 for x in mine)
        print(f"  effort={str(lvl or 'none'):<7} ok {len(ok)}/{len(mine)}  "
              f"reasoning {st.median(rt) if rt else 0:>6.0f}  "
              f"output {st.median(ct) if ct else 0:>6.0f}  "
              f"lat {st.median(lat) if lat else 0:>5.1f}s  ${cost:.4f}")

    print("\n  If the reasoning column is flat across all four rows, the")
    print("  provider is ignoring the parameter and ablation A5 is dead for")
    print("  this model — the cost has to come from frames or resolution.")
    print("  If it moves, A5 is the cheapest 36-53% on the table.")


if __name__ == "__main__":
    main()
