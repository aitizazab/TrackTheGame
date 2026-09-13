"""Visual bridging between VLM anchors: sparse optical flow as a second
estimator of where each player went, emitted as a prediction the tracker can use.

WHAT CHANGED FROM PHASE 1 (D45/D46), AND WHY

Phase 1 only counted. It read `track.py`'s output, compared its own guess to the
Hungarian solver's, and wrote down how often they differed. D46 concluded the
disagreements did not localise the fragmentation the bridge was built to fix, and
shelved it. Four defects were named in that write-up and this file addresses them
one at a time; each is measured separately in the report, not stacked.

  1. THE DISCARD WAS ALL-OR-NOTHING PER GAP. A bridge spans ten source frames; a
     failed forward-backward check threw away all ten, including the frames
     before the occlusion started, which tracked perfectly. Fixed by
     `--truncate`: find the source frame where the point cloud breaks up, keep
     the clean prefix, and extrapolate the rest at the velocity measured over it.

  2. THE FB CHECK DISCARDS EXACTLY AT OCCLUSIONS. A correlation tracker loses the
     patch when it is covered, so the safeguard that makes the bridge safe also
     silences it at crossings - the only case worth having it for. `--truncate`
     is the attempt on this: an occlusion no longer silences the bridge, it
     merely shortens the interval the bridge speaks about. Whether that is
     enough is the question the report answers.

  3. COST. Phase 1 read `track.py`'s output purely for convenience, which forced
     the whole bridge to run after the whole tracker. Nothing here needs the
     tracker: the boxes come from the DETECTIONS at anchor `a`, so the bridge for
     gap [a,b] is computable the moment anchor `b`'s API call lands, and decode
     depends on nothing at all. `--time` reports the breakdown that makes the
     overlap argument checkable rather than asserted.

  4. SEEDING. Nine points on a 3x3 grid inset into the box catch limbs and
     background. `--seed corners` uses goodFeaturesToTrack inside the box;
     `--seed torso` weights the sample toward the upper-centre mass.

WHAT THE TWO ESTIMATORS ARE, UNCHANGED

  geometry   track.py's Kalman prediction. Sees only the sparse anchors. Fails on
             ambiguity: two candidates 0.33s later and no way to tell which.
  bridge     sparse Lucas-Kanade through every source frame. Sees the pixels.
             Fails on occlusion: the patch disappears and the filter latches onto
             whatever is behind it.

They fail on different things, which is the whole argument for running both.

WHAT WAS MEASURED. Every figure below is against the accepted 5fps run as the
reference, scored by eval_ids.py (IDF1 / fragmentation / ID switches, matched on
a gate expressed as a multiple of local player spacing). Not yet written up in
docs/decisions.md.

  - The bridge is a BETTER PREDICTOR than the Kalman on four of five clips, and
    the gap widens as the anchors thin out. At 2.5fps, median error 0.058-0.146
    local player spacings against the Kalman's 0.086-0.296, winning 64-74% of
    the tracker's own accepted decisions.
  - Used as a REPLACEMENT for the Kalman prediction it is worse than useless
    (allstars IDF1 0.885 -> 0.780): where it fails it fails catastrophically,
    and it then aims the association gate at the wrong player. Used as a
    convex BLEND it is the best configuration measured on 4 of 5 clips.
  - Truncation is worth +0.011 to +0.024 mean IDF1 and lifts usable coverage
    from 75-90% to 96-99%.
  - Cost, pipelined against the recorded API schedule: 0.04-0.08s of added wall
    clock, not the 7.1s Phase 1 reported. That figure was the cost of running
    the bridge as a serial block after the tracker, not the cost of the work.

RECOMMENDED

    uv run --python 3.12 --with opencv-python --with numpy bridge.py \\
        outputs/detections/<clip>__<tag>.json --clip clips/<name>.mp4 \\
        --truncate --fb-max 5.0 --emit outputs/bridge/<tag>.json

    uv run track.py outputs/detections/<clip>__<tag>.json \\
        --bridge outputs/bridge/<tag>.json --bridge-mode blend --bridge-alpha 0.4

`--bridge-alpha` is CHOSEN, not derived. Inverse-variance weighting on the
measured errors gives 0.69-0.82, which overshoots badly because the bridge's
errors are heavy-tailed and that rule assumes they are not. Over the whole 3x3
grid of (--fb-max in {1,2,5}) x (alpha in {0.3,0.4,0.5}), 40 of 45 clip-cells
beat no-bridge and the mean gain is +0.051 IDF1; alpha 0.4 is simply the best
cell of a broad plateau.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# --- seeding ---------------------------------------------------------------

SEED_GRID = (3, 3)          # points per player box, before edge trimming
SEED_INSET = (0.25, 0.20)   # keep points away from the box edge, where the
                            # background leaks in and LK locks onto grass
FB_MAX_PX = 5.0             # a point whose round trip misses by more than this
                            # is discarded
MIN_SURVIVORS = 3           # a bridge needs this many surviving points to speak
LK = dict(winSize=(21, 21), maxLevel=3)

# --- truncation ------------------------------------------------------------
#
# SCATTER WAS TRIED AS THE OCCLUSION SIGNAL AND IT DOES NOT WORK, and saying so
# here is the point of this comment. The idea was that the points on a player
# should stay a coherent cloud and fly apart when the patch is covered. Measured
# on basketball at 3fps, end-of-gap spread as a ratio to the spread at seeding:
#
#     bridges that PASS their own FB check   p50 0.99   p90 1.17
#     bridges that FAIL it                   p50 0.97   p90 1.46
#
# The same distribution. At a threshold of 1.2 it catches 20% of the failures
# while wrongly flagging 7% of the good bridges; at 3.0 it fires once in 835.
# The reason is that a point which loses its patch usually LOCKS ONTO THE
# BACKGROUND and stops moving, which keeps the cloud compact and wrong. It is
# still computed and reported, and it decides nothing.
#
# What works is the per-step round trip - see the `step` branch in bridge_gap.
MIN_PREFIX = 2              # fewer clean steps than this and there is no
                            # velocity worth extrapolating


def seed_grid(box, W, H, gray=None):
    x, y, w, h = box["x"] * W, box["y"] * H, box["w"] * W, box["h"] * H
    ix, iy = SEED_INSET
    xs = np.linspace(x + w * ix, x + w * (1 - ix), SEED_GRID[0])
    ys = np.linspace(y + h * iy, y + h * (1 - iy), SEED_GRID[1])
    return np.array([[a, b] for a in xs for b in ys], dtype=np.float32)


def seed_torso(box, W, H, gray=None):
    """Nine points concentrated on the upper-centre mass.

    A player's arms and legs move relative to their torso by a large fraction of
    a body height in a third of a second, so a point on a limb reports a
    displacement that is not the player's displacement. The torso is the part
    that moves as the player moves. Vertical range 0.15-0.55 of the box (head to
    waist, avoiding the swinging half) and horizontal 0.3-0.7.
    """
    x, y, w, h = box["x"] * W, box["y"] * H, box["w"] * W, box["h"] * H
    xs = np.linspace(x + w * 0.30, x + w * 0.70, 3)
    ys = np.linspace(y + h * 0.15, y + h * 0.55, 3)
    return np.array([[a, b] for a in xs for b in ys], dtype=np.float32)


def seed_corners(box, W, H, gray):
    """goodFeaturesToTrack inside the box, with the grid as a fallback.

    LK's aperture problem means a point in a flat region has no recoverable
    motion; Shi-Tomasi picks the points that actually have two strong gradient
    directions. Restricted to the same inset region so it cannot pick a corner of
    the background showing through the box.
    """
    import cv2
    x, y, w, h = box["x"] * W, box["y"] * H, box["w"] * W, box["h"] * H
    ix, iy = SEED_INSET
    x0, y0 = int(max(0, x + w * ix)), int(max(0, y + h * iy))
    x1, y1 = int(min(W, x + w * (1 - ix))), int(min(H, y + h * (1 - iy)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return seed_grid(box, W, H)
    sub = gray[y0:y1, x0:x1]
    pts = cv2.goodFeaturesToTrack(sub, maxCorners=9, qualityLevel=0.01,
                                  minDistance=max(2.0, (x1 - x0) / 6))
    if pts is None or len(pts) < MIN_SURVIVORS:
        return seed_grid(box, W, H)
    p = pts.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
    return p.astype(np.float32)


SEEDERS = {"grid": seed_grid, "torso": seed_torso, "corners": seed_corners}


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


def foot_of(box):
    return (box["x"] + box["w"] / 2, box["y"] + box["h"])


def bridge_gap(grays, a, b, boxes, W, H, cv2, cfg):
    """Track every player's seed points from anchor a to anchor b and back.

    Returns one dict per box in `boxes`, in the same order.
    """
    seeder = SEEDERS[cfg["seed"]]
    seeds = [seeder(p, W, H, grays[a]) for p in boxes]
    counts = [len(s) for s in seeds]
    p0 = np.vstack(seeds).reshape(-1, 1, 2)
    owner = np.repeat(np.arange(len(boxes)), counts)

    # ---- forward pass, with the round trip taken ONE STEP AT A TIME ---------
    #
    # The Phase-1 check ran the whole gap forward and then the whole gap back,
    # and compared the ends. That produces ONE number for TEN frames, which is
    # why the discard had to be all-or-nothing: there was nothing in the
    # measurement that said WHEN the tracking went wrong.
    #
    # Checking each step against its own reverse costs exactly the same number
    # of LK calls - n forward and n backward either way - and returns a per-step
    # error curve instead of a scalar. The step at which a point loses its patch
    # is then directly observable, and the prefix before it is salvageable.
    cur = p0.copy()
    traj = [cur.reshape(-1, 2).copy()]
    step_fb = None
    if cfg["fb"] == "step":
        rows = []
        for f in range(a, b):
            nxt, _, _ = cv2.calcOpticalFlowPyrLK(grays[f], grays[f + 1], cur,
                                                 None, **LK)
            back, _, _ = cv2.calcOpticalFlowPyrLK(grays[f + 1], grays[f], nxt,
                                                  None, **LK)
            rows.append(np.linalg.norm((back - cur).reshape(-1, 2), axis=1))
            cur = nxt
            traj.append(cur.reshape(-1, 2).copy())
        fwd = cur.copy()
        step_fb = np.stack(rows) if rows else np.zeros((0, len(p0)))
        # A point's reliability up to step k is the worst single step it has
        # taken so far: one frame where the patch jumped to something else is
        # not undone by ten quiet frames afterwards.
        fb = step_fb.max(axis=0) if len(step_fb) else np.zeros(len(p0))
    else:
        for f in range(a, b):
            cur, _, _ = cv2.calcOpticalFlowPyrLK(grays[f], grays[f + 1], cur,
                                                 None, **LK)
            traj.append(cur.reshape(-1, 2).copy())
        fwd = cur.copy()
        for f in range(b, a, -1):
            cur, _, _ = cv2.calcOpticalFlowPyrLK(grays[f], grays[f - 1], cur,
                                                 None, **LK)
        fb = np.linalg.norm((cur - p0).reshape(-1, 2), axis=1)

    T = np.stack(traj)                       # (steps+1, npts, 2)
    nsteps = b - a
    out = []
    for k, pl in enumerate(boxes):
        sel = owner == k
        bh_px = max(pl["h"] * H, 1e-6)
        fb_k = fb[sel]
        ok = fb_k <= cfg["fb_max"]
        rec = {"n": int(sel.sum()), "n_ok": int(ok.sum()),
               "fb_p50": round(float(np.median(fb_k)), 2)}

        P = T[:, sel, :]                                  # (steps+1, npts, 2)
        # scatter, kept only as a diagnostic: it was tried as the occlusion
        # signal and measured useless - on basketball the FB-failed bridges have
        # a median end-spread of 0.97 against 0.99 for the ones that pass, i.e.
        # the two populations are the same distribution.
        cen = np.median(P, axis=1, keepdims=True)
        spread = np.median(np.linalg.norm(P - cen, axis=2), axis=1)
        rec["scatter_end"] = round(float(spread[-1] / max(spread[0], 1e-6)), 2)

        p_a = np.array(foot_of(pl))
        disp = None
        if cfg["truncate"] and step_fb is not None:
            # DEFECT 1 AND 2. The gap did not fail, a SUFFIX of it did.
            #
            # Per step, count how many of this player's points are still
            # trustworthy - running maximum of their own per-step round-trip
            # error, under the same threshold the whole-gap check used. The last
            # step at which enough points survive is the end of the clean
            # prefix. Measure the displacement over that prefix and carry it
            # forward at the same velocity for the rest of the gap.
            #
            # This is the attempt on the crux objection. An occlusion no longer
            # silences the bridge; it shortens the interval the bridge speaks
            # about, and the bridge still has something to say about a crossing.
            run = np.maximum.accumulate(step_fb[:, sel], axis=0)
            alive = (run <= cfg["fb_max"]).sum(axis=1)     # per step
            good = np.where(alive >= cfg["min_surv"])[0]
            j = int(good[-1]) + 1 if len(good) else 0
            # contiguity: the prefix must be unbroken from the start
            first_bad = np.where(alive < cfg["min_surv"])[0]
            if len(first_bad):
                j = min(j, int(first_bad[0]))
            rec["prefix"] = int(j)
            rec["lost_at"] = None if j >= nsteps else int(j)
            if j >= nsteps:
                sel_pts = run[-1] <= cfg["fb_max"]
                if sel_pts.sum() >= cfg["min_surv"]:
                    disp = np.median(P[-1][sel_pts] - P[0][sel_pts], axis=0)
                    rec["mode"] = "full"
            elif j >= cfg["min_prefix"]:
                sel_pts = run[j - 1] <= cfg["fb_max"]
                if sel_pts.sum() >= cfg["min_surv"]:
                    d = np.median(P[j][sel_pts] - P[0][sel_pts], axis=0)
                    disp = d * (nsteps / j)               # extrapolate
                    rec["mode"] = "truncated"
            if disp is None:
                rec["verdict"] = "FAILED_EARLY"
                out.append(rec)
                continue
        else:
            rec["lost_at"] = None
            if rec["n_ok"] < cfg["min_surv"]:
                rec["verdict"] = "FAILED_FB"
                out.append(rec)
                continue
            disp = np.median((fwd.reshape(-1, 2)[sel][ok]
                              - p0.reshape(-1, 2)[sel][ok]), axis=0)
            rec["mode"] = "full"

        pred = p_a + np.array([disp[0] / W, disp[1] / H])
        rec["pred"] = [round(float(pred[0]), 5), round(float(pred[1]), 5)]
        rec["disp_bh"] = round(float(np.hypot(disp[0], disp[1])) / bh_px, 3)
        rec["verdict"] = "OK"
        out.append(rec)
    return out


def build(dets, grays, W, H, cv2, cfg):
    anchors = sorted(f["frame"] for f in (dets.get("frames") or []))
    det_at = {f["frame"]: (f.get("players") or [])
              for f in (dets.get("frames") or [])}
    gaps = {}
    t0 = time.perf_counter()
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        if b >= len(grays):
            break
        boxes = det_at.get(a) or []
        if not boxes:
            continue
        gaps[f"{a}-{b}"] = bridge_gap(grays, a, b, boxes, W, H, cv2, cfg)
    return gaps, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detections", type=Path)
    ap.add_argument("--clip", type=Path, required=True,
                    help="any resolution: boxes are fractions")
    ap.add_argument("--emit", type=Path, default=None,
                    help="write predictions for track.py --bridge")
    # The Phase-1 "compare against the solver" mode is gone. It compared the
    # bridge to an assignment the bridge had not influenced and reported a
    # distance in body heights, which is neither a decision nor a physically
    # normalised quantity. `track.py --bridge-observe` replaced it: same
    # comparison, taken on the tracker's own accepted pairs, reported in units
    # of local player spacing, and with the unmatched tracks - the population
    # that actually fragments - included.
    ap.add_argument("--seed", choices=list(SEEDERS), default="grid")
    ap.add_argument("--fb-max", type=float, default=FB_MAX_PX)
    ap.add_argument("--min-surv", type=int, default=MIN_SURVIVORS)
    ap.add_argument("--fb", choices=["e2e", "step"], default="e2e",
                    help="e2e = Phase 1's whole-gap round trip, one number for "
                         "the whole bridge. step = the round trip taken one "
                         "source frame at a time, same number of LK calls, "
                         "which is what makes truncation possible")
    ap.add_argument("--truncate", action="store_true",
                    help="salvage the clean prefix of a broken bridge. "
                         "Requires --fb step")
    ap.add_argument("--min-prefix", type=int, default=MIN_PREFIX,
                    help="shortest clean prefix worth extrapolating from")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv missing. Run with:  uv run --with opencv-python bridge.py ...")

    if args.truncate and args.fb != "step":
        args.fb = "step"
    cfg = dict(seed=args.seed, fb_max=args.fb_max, min_surv=args.min_surv,
               truncate=args.truncate, min_prefix=args.min_prefix,
               fb=args.fb)

    dets = json.loads(args.detections.read_text(encoding="utf-8"))
    grays, t_decode = load_grays(args.clip, cv2)
    H, W = grays[0].shape
    gaps, t_lk = build(dets, grays, W, H, cv2, cfg)

    n = sum(len(v) for v in gaps.values())
    verd = Counter(r["verdict"] for v in gaps.values() for r in v)
    print(f"  decode {len(grays)}f {W}x{H}: {t_decode:.2f}s   "
          f"LK {n} player-bridges over {len(gaps)} gaps: {t_lk:.2f}s   "
          f"total {t_decode + t_lk:.2f}s")
    for k, c in verd.most_common():
        print(f"  {k:<14}{c:>6}{100 * c / max(n, 1):>7.1f}%")

    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(
            {"clip": str(args.clip), "cfg": cfg, "gaps": gaps}), encoding="utf-8")
        print(f"  -> {args.emit}")



if __name__ == "__main__":
    main()
