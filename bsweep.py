"""One bridge configuration at a time: build it, score the PREDICTOR alone.

Deliberately does not run the tracker with the bridge switched on. Changing the
association changes the history, and a metric taken after that cannot say
whether the predictor got better or the run merely diverged. So every row here
is a paired comparison on the SHIPPED tracker's own decisions - identical
history, identical detections, two predictors scored against the same answer.

  brg/nn p50   median prediction error in units of local player spacing
  wins         share of accepted pairs where the bridge landed closer
  cover        share of accepted pairs for which a bridge existed at all
               (a bridge that grades itself unusable predicts nothing, and a
               predictor that is silent cannot be better)
  rescue       unmatched track-frames the bridge reaches and the Kalman does not
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path("outputs/bridge/_sweep")


def build(dets, clip, flags, tag):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / f"{tag}.json"
    cmd = [sys.executable, "bridge.py", dets, "--clip", clip,
           "--emit", str(out)] + flags
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit(f"bridge.py failed for {tag}")
    g = json.loads(out.read_text(encoding="utf-8"))["gaps"]
    rec = [x for v in g.values() for x in v]
    okshare = sum(1 for x in rec if x["verdict"] == "OK") / max(len(rec), 1)
    # bridge.py prints its own timing; capture the LK seconds
    lk = 0.0
    for line in r.stdout.splitlines():
        if "LK " in line and "gaps:" in line:
            lk = float(line.split("gaps:")[1].split("s")[0])
    return out, okshare, lk


def observe(dets, bridge, tag):
    rows = SCRATCH / f"{tag}.rows.jsonl"
    canon = Path("outputs/tracks") / (Path(dets).stem + "__tracks.json")
    backup = canon.read_bytes() if canon.exists() else None
    cmd = [sys.executable, "track.py", dets, "--bridge", str(bridge),
           "--bridge-observe", "--bridge-rows", str(rows)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if backup is not None:
        canon.write_bytes(backup)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("track.py failed")
    return [json.loads(l) for l in rows.open(encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    print(f"  {'config':<24}{'OK%':>7}{'cover':>8}{'kal/nn':>9}{'brg/nn':>9}"
          f"{'p90 kal':>9}{'p90 brg':>9}{'wins':>7}{'resc':>6}{'LKs':>7}")
    for tag, flags in spec:
        out, okshare, lk = build(args.dets, args.clip, flags, tag)
        rows = observe(args.dets, out, tag)
        m = [r for r in rows if not r.get("unmatched")]
        u = [r for r in rows if r.get("unmatched")]
        if not m:
            print(f"  {tag:<24} no paired rows")
            continue
        k = np.array([r["kal"] for r in m])
        b = np.array([r["brg"] for r in m])
        nn = np.array([r["nn"] for r in m])
        rescue = 0
        if u:
            uk = np.array([r["kal_near"] for r in u])
            ub = np.array([r["brg_near"] for r in u])
            ug = np.array([r["gate"] for r in u])
            rescue = int(((ub <= ug) & (uk > ug)).sum())
        print(f"  {tag:<24}{100 * okshare:>6.1f}%{len(m):>8}"
              f"{np.median(k / nn):>9.3f}{np.median(b / nn):>9.3f}"
              f"{np.percentile(k / nn, 90):>9.3f}{np.percentile(b / nn, 90):>9.3f}"
              f"{100 * (b < k).mean():>6.1f}%{rescue:>6}{lk:>7.2f}")


if __name__ == "__main__":
    main()
