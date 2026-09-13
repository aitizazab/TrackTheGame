"""What the bridge actually adds to the WALL CLOCK, measured rather than argued.

Phase 1 reported "7.1s added" by timing decode plus LK as a block that runs
after the tracker. That is the cost of the convenience of reading track.py's
output, not the cost of the work. Two facts make it avoidable:

  - decode depends on nothing. It can start at t=0, alongside the first API
    call.
  - the bridge for gap [a,b] needs the frames a..b and the DETECTIONS at a.
    Nothing from the tracker. So it can run the moment anchor b's call lands.

And the governing fact from HANDOFF: all 150 calls dispatch inside 0.66s and run
concurrently, so **wall clock is set by the slowest call**, not the median. Any
CPU work that finishes before that call lands is free.

This harness does the real work on the real clip, and releases each anchor at
the time that anchor's API call actually returned in the recorded run
(`t_done_s` in docs/run_log.jsonl). No network, no spend - the schedule is
replayed from the log. What it reports is the difference between the time the
last anchor lands and the time the last bridge finishes, which is the only
number that can honestly be called "added wall clock".
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

import bridge as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--log", default="docs/run_log.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--fb-max", type=float, default=2.0)
    ap.add_argument("--seed", default="grid")
    args = ap.parse_args()

    import cv2

    dets = json.loads(Path(args.dets).read_text(encoding="utf-8"))
    det_at = {f["frame"]: (f.get("players") or []) for f in dets["frames"]}
    anchors = sorted(det_at)

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8")
            if f'"{args.tag}"' in l]
    done = {}
    for r in rows:
        if r.get("tag") == args.tag and r.get("ok") and r.get("t_done_s"):
            done[r["frame"]] = float(r["t_done_s"])
    if not done:
        raise SystemExit(f"no logged calls for tag {args.tag}")
    last_call = max(done.values())
    print(f"  recorded run: {len(done)} calls, last landed at "
          f"t+{last_call:.2f}s  (this IS the no-bridge wall clock)")

    cfg = dict(seed=args.seed, fb_max=args.fb_max, min_surv=B.MIN_SURVIVORS,
               truncate=args.truncate, min_prefix=B.MIN_PREFIX,
               fb="step" if args.truncate else "e2e")

    grays = []
    decode_done = threading.Event()
    t0 = time.perf_counter()

    def decoder():
        cap = cv2.VideoCapture(args.clip)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        cap.release()
        decode_done.set()

    th = threading.Thread(target=decoder, daemon=True)
    th.start()

    # Consumer: walk the gaps in order, wait until (a) both anchors have landed
    # in the replayed schedule and (b) the frames are decoded, then do the LK.
    # READINESS ORDER, not anchor order. Calls dispatch concurrently and land
    # out of order, so a queue that insists on gap 3 before gap 4 blocks behind
    # whichever call happened to be slow. A real implementation takes whatever
    # is ready; walking the gaps in index order was costing 1.6s of pure
    # self-inflicted head-of-line blocking.
    finished_at = {}
    todo = []
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        if a not in done or b not in done:
            continue
        todo.append((max(done[a], done[b]), a, b))
    todo.sort()
    for ready, a, b in todo:
        while time.perf_counter() - t0 < ready:
            time.sleep(0.002)
        while len(grays) <= b and not decode_done.is_set():
            time.sleep(0.002)
        if b >= len(grays):
            continue
        boxes = det_at[a]
        if not boxes:
            continue
        H, W = grays[0].shape
        B.bridge_gap(grays, a, b, boxes, W, H, cv2, cfg)
        finished_at[(a, b)] = time.perf_counter() - t0

    th.join()
    end = max(finished_at.values()) if finished_at else 0.0
    lag = np.array([finished_at[k] - max(done[k[0]], done[k[1]])
                    for k in finished_at])
    print(f"  last bridge finished at t+{end:.2f}s")
    print(f"  ADDED WALL CLOCK: {max(0.0, end - last_call):.2f}s")
    print(f"  per-gap lag behind its own anchor: p50 {np.median(lag):.3f}s  "
          f"p90 {np.percentile(lag, 90):.3f}s  max {lag.max():.3f}s")


if __name__ == "__main__":
    main()
