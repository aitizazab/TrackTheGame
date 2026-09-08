"""
Phase 1 of the visual reconciliation stage: OBSERVE ONLY.

This module changes nothing. It builds a second, independent estimate of where
each track went between two VLM anchors, grades that estimate against itself,
compares it to what the Hungarian solver decided, and writes down how often the
two disagree and in what way. No pipeline output is altered.

WHY OBSERVE FIRST. Twice in this project a mechanism was wired in before its
base rate was measured, and both times it made things worse: the two-way ball
outlier test (indicted the truth when decoys locally outnumbered real
detections) and the round-trip reachability exemption (built from a ceiling, so
it spared a decoy). Both would have been caught by a pass that only counted.

WHAT THE TWO ESTIMATORS ARE

  geometry   track.py's Kalman prediction + Hungarian assignment. Sees only the
             sparse anchor detections. Fails on ambiguity: at a crossing, two
             candidates 0.33s later and no way to tell which is which.

  bridge     sparse Lucas-Kanade optical flow, seeded from the anchor box and
             run through every SOURCE frame to the next anchor. Sees the actual
             pixels. Fails on occlusion: the tracked patch disappears behind the
             occluder and the filter latches onto whatever is there.

They fail on different things, which is the entire argument for running both.

THE ASYMMETRY THAT MAKES ADJUDICATION POSSIBLE. A drifting visual tracker fails
*smoothly*, and every guard in track.py keys on discontinuity - the residual
gate, the round-trip test, the outlier check all ask "did this jump?". Smooth
wrongness would pass all of them. What saves it is that the bridge can grade
ITSELF: track each point forward to the next anchor and then backward again, and
measure how far it returns from where it started. Measured on allstars at 3fps,
1547 player-bridges: forward-backward error p50 0.29px, p90 6.42px, 12.2% above
5px. Most bridges are near-exact and the failures are a separable tail, not a
smear - which is the distribution a discard rule needs.

So the bridge arrives carrying an error bar. The Hungarian assignment does not.

USAGE

    uv run --with opencv-python bridge.py \\
        outputs/tracks/allstars_fr_eng_1080__allstars_3fps__tracks.json \\
        --clip clips/allstars_fr_eng.mp4 \\
        --detections outputs/detections/allstars_fr_eng_1080__allstars_3fps.json

Writes docs/bridge_log.jsonl (one row per player-bridge) and prints a summary.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# --- tuning, all of it provisional and none of it acted on in Phase 1 -------

SEED_GRID = (3, 3)          # points per player box, before edge trimming
SEED_INSET = (0.25, 0.20)   # keep points away from the box edge, where the
                            # background leaks in and LK locks onto grass
FB_MAX_PX = 5.0             # a point whose round trip misses by more than this
                            # is discarded. 12.2% of points exceed it, and they
                            # are the tail, not the body, of the distribution
MIN_SURVIVORS = 3           # a bridge needs this many surviving points to vote
AGREE_BH = 0.5              # prediction within this many body heights of the
                            # solver's answer counts as agreement. Deliberately
                            # generous: we are asking "same player?", not
                            # "same pixel?"
LK = dict(winSize=(21, 21), maxLevel=3)


def seed_points(box, W, H):
    """A small grid inside the box, inset from the edges."""
    x, y, w, h = box["x"] * W, box["y"] * H, box["w"] * W, box["h"] * H
    ix, iy = SEED_INSET
    xs = np.linspace(x + w * ix, x + w * (1 - ix), SEED_GRID[0])
    ys = np.linspace(y + h * iy, y + h * (1 - iy), SEED_GRID[1])
    return np.array([[a, b] for a in xs for b in ys], dtype=np.float32).reshape(-1, 1, 2)


def load_grays(clip, cv2):
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(clip))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not out:
        sys.exit(f"could not read any frames from {clip}")
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path, help="a __tracks.json from track.py")
    ap.add_argument("--clip", type=Path, required=True,
                    help="the RENDER clip (720p). Boxes are fractions, so the "
                         "resolution only has to be self-consistent")
    ap.add_argument("--detections", type=Path, required=True,
                    help="the detections json, for the raw anchor frames")
    ap.add_argument("--fb-max", type=float, default=FB_MAX_PX)
    ap.add_argument("--out", type=Path, default=Path("docs/bridge_log.jsonl"))
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv missing. Run with:  uv run --with opencv-python bridge.py ...")

    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))
    dets = json.loads(args.detections.read_text(encoding="utf-8"))

    # Anchors are the frames the model actually answered.
    anchors = sorted(f["frame"] for f in (dets.get("frames") or []))
    det_at = {f["frame"]: (f.get("players") or []) for f in (dets.get("frames") or [])}
    # track.py's output is interpolated to every source frame; at an anchor
    # frame the drawn position is the detection it was assigned.
    drawn = {f["frame"]: (f.get("players") or []) for f in (tracks.get("frames") or [])}

    grays, t_decode = load_grays(args.clip, cv2)
    H, W = grays[0].shape
    print(f"  decoded {len(grays)} frames at {W}x{H} in {t_decode:.2f}s")
    print(f"  {len(anchors)} anchors, {len(anchors) - 1} gaps to bridge")

    rows, verdicts = [], Counter()
    fb_all = []
    t0 = time.perf_counter()

    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        if b >= len(grays):
            break
        here = drawn.get(a) or []
        if not here:
            continue

        # one LK call for every point of every player in this gap
        p0 = np.vstack([seed_points(p, W, H) for p in here])
        owner = np.repeat(np.arange(len(here)), SEED_GRID[0] * SEED_GRID[1])

        cur = p0.copy()
        for f in range(a, b):
            cur, _, _ = cv2.calcOpticalFlowPyrLK(grays[f], grays[f + 1], cur, None, **LK)
        fwd = cur.copy()
        for f in range(b, a, -1):
            cur, _, _ = cv2.calcOpticalFlowPyrLK(grays[f], grays[f - 1], cur, None, **LK)
        fb = np.linalg.norm((cur - p0).reshape(-1, 2), axis=1)
        fb_all.append(fb)

        there = {p["track"]: p for p in (drawn.get(b) or [])}
        dets_b = det_at.get(b) or []

        for k, pl in enumerate(here):
            sel = (owner == k) & (fb <= args.fb_max)
            n_ok = int(sel.sum())
            row = {"gap": [a, b], "track": pl["track"], "label": pl.get("label"),
                   "survivors": n_ok, "fb_p50": round(float(np.median(fb[owner == k])), 2)}

            if n_ok < MIN_SURVIVORS:
                row["verdict"] = "BRIDGE_FAILED"
                verdicts["BRIDGE_FAILED"] += 1
                rows.append(row)
                continue

            # median displacement of the surviving points -> predicted centre
            d = (fwd.reshape(-1, 2)[sel] - p0.reshape(-1, 2)[sel])
            pred = np.array([(pl["x"] + pl["w"] / 2) * W, (pl["y"] + pl["h"] / 2) * H]) \
                + np.median(d, axis=0)
            bh = max(pl["h"] * H, 1e-6)          # body height, in pixels
            row["pred"] = [round(float(pred[0]) / W, 4), round(float(pred[1]) / H, 4)]

            # what did the solver say?
            solver = there.get(pl["track"])
            # what is the nearest raw detection to the bridge's prediction?
            best, best_d = None, 1e9
            for j, dt in enumerate(dets_b):
                c = np.array([(dt["x"] + dt["w"] / 2) * W, (dt["y"] + dt["h"] / 2) * H])
                dist = float(np.linalg.norm(c - pred)) / bh
                if dist < best_d:
                    best, best_d = j, dist
            row["nearest_det_bh"] = round(best_d, 3) if best is not None else None

            if solver is None:
                # the solver lost this track across the gap
                row["verdict"] = "SOLVER_LOST" if best_d < AGREE_BH else "BOTH_LOST"
            else:
                sc = np.array([(solver["x"] + solver["w"] / 2) * W,
                               (solver["y"] + solver["h"] / 2) * H])
                gap_bh = float(np.linalg.norm(sc - pred)) / bh
                row["solver_vs_bridge_bh"] = round(gap_bh, 3)
                if gap_bh <= AGREE_BH:
                    row["verdict"] = "AGREE"
                elif best_d < gap_bh:
                    row["verdict"] = "DISAGREE_OTHER_DET"   # possible swap
                else:
                    row["verdict"] = "DISAGREE_NO_DET"      # bridge drifted, probably
            verdicts[row["verdict"]] += 1
            rows.append(row)

    t_track = time.perf_counter() - t0
    fb_all = np.concatenate(fb_all) if fb_all else np.array([0.0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    total = sum(verdicts.values()) or 1
    print(f"\n  bridged {total} player-gaps in {t_track:.2f}s "
          f"(+{t_decode:.2f}s decode = {t_decode + t_track:.2f}s added)")
    print(f"  forward-backward error px: p50 {np.median(fb_all):.2f}  "
          f"p90 {np.percentile(fb_all, 90):.2f}  above {args.fb_max}px "
          f"{100 * (fb_all > args.fb_max).mean():.1f}%")
    print(f"\n  {'verdict':<20}{'n':>6}{'share':>9}")
    order = ["AGREE", "DISAGREE_OTHER_DET", "DISAGREE_NO_DET",
             "SOLVER_LOST", "BOTH_LOST", "BRIDGE_FAILED"]
    for v in order:
        if verdicts.get(v):
            print(f"  {v:<20}{verdicts[v]:>6}{100 * verdicts[v] / total:>8.1f}%")
    print(f"\n  -> {args.out}")
    print("\n  Phase 1 changes nothing. Read the disagreement rate before "
          "deciding any policy.")


if __name__ == "__main__":
    main()
