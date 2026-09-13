"""Identity quality of one tracks.json measured against another, taken as reference.

WHY THIS EXISTS. Every identity metric this project has used counts events - how
many labels were drawn, how many jersey numbers were read - and D42's own note
says that is exactly the failure mode: "marker COUNT matched while marker
PLACEMENT did not". Label count is also blind in the direction that matters: on
basketball 3fps and 5fps both draw 21 identities, so by that measure the 3fps
deficit does not exist, which cannot be right.

WHAT IS MEASURED. The accepted 5fps render is the reference. At every SOURCE
frame both runs place a set of feet. Match them, accumulate a co-occurrence
matrix between reference identity and test identity, then solve ONE global
one-to-one assignment between the two identity sets that maximises matched
frames. That is the standard ID-measure (Ristani et al.) and it is the right
shape for this question: a reference track that the test run splits in two can
only claim one of the halves, so fragmentation shows up directly as lost IDF1,
whereas a count of labels cannot see it.

  IDTP   frames where ref identity r and its assigned test identity t coincide
  IDFN   frames where r is drawn and its assigned t is not there
  IDFP   frames where t is drawn and its assigned r is not there
  IDF1   2*IDTP / (2*IDTP + IDFP + IDFN)

Also reported, because IDF1 mixes detection errors with identity errors:

  frag   for each reference track, how many DISTINCT test tracks cover it
         (weighted by frames). 1.00 = never split. This isolates identity.
  IDSW   how often the test id matched to a reference track changes

THE MATCHING GATE IS THE ONE THRESHOLD AND IT IS NORMALISED, NOT CHOSEN.
Two feet are the same player if they are closer than the distance to the nearest
OTHER player in that frame - below that separation the two estimates provably
cannot be pointing at different people. So the gate is a multiple of the local
nearest-neighbour spacing, and `--gate-sweep` reports every metric across a range
of that multiple so the conclusion can be shown not to depend on it.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def foot(p):
    """Where the marker is drawn: bottom-centre of the box."""
    return (p["x"] + p["w"] / 2, p["y"] + p["h"])


def load(path):
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    return {f["frame"]: (f.get("players") or []) for f in (t.get("frames") or [])}, t


def frame_match(ref, test, gate_mul, aspect=16 / 9):
    """One frame. Returns list of (ref_id, test_id) pairs.

    Distances are computed with x scaled by the frame aspect so that a fraction
    of the width and a fraction of the height are the same physical distance;
    otherwise a 16:9 frame makes horizontal separation look 1.78x smaller than it
    is, and the spacing normalisation inherits the error.
    """
    if not ref or not test:
        return []
    R = np.array([foot(p) for p in ref]) * np.array([aspect, 1.0])
    T = np.array([foot(p) for p in test]) * np.array([aspect, 1.0])

    # local spacing: for each reference foot, distance to the nearest OTHER
    # reference foot. This is a property of the scene, not of either estimator.
    if len(R) >= 2:
        dd = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=2)
        np.fill_diagonal(dd, 1e9)
        nn = dd.min(axis=1)
    else:
        nn = np.array([1e9])

    cost = np.linalg.norm(R[:, None, :] - T[None, :, :], axis=2)
    gate = gate_mul * nn[:, None]
    big = 1e6
    c = np.where(cost <= gate, cost, big)
    ri, ci = linear_sum_assignment(c)
    return [(ref[i]["id"], test[j]["id"])
            for i, j in zip(ri, ci) if c[i, j] < big]


def evaluate(ref_frames, test_frames, gate_mul):
    frames = sorted(set(ref_frames) | set(test_frames))
    co = defaultdict(int)               # (ref_id, test_id) -> frames matched
    ref_n, test_n = defaultdict(int), defaultdict(int)
    per_ref_seq = defaultdict(list)     # ref_id -> [test_id per matched frame]

    for f in frames:
        r = ref_frames.get(f) or []
        t = test_frames.get(f) or []
        for p in r:
            ref_n[p["id"]] += 1
        for p in t:
            test_n[p["id"]] += 1
        for a, b in frame_match(r, t, gate_mul):
            co[(a, b)] += 1
            per_ref_seq[a].append(b)

    R = sorted(ref_n)
    T = sorted(test_n)
    if not R or not T:
        return None
    ri = {k: i for i, k in enumerate(R)}
    ti = {k: i for i, k in enumerate(T)}

    # global one-to-one identity assignment maximising matched frames.
    # Pad to a square problem so unmatched identities are allowed.
    n = len(R) + len(T)
    M = np.zeros((n, n))
    for (a, b), v in co.items():
        M[ri[a], ti[b]] = v
    rr, cc = linear_sum_assignment(-M)
    idtp = int(sum(M[i, j] for i, j in zip(rr, cc)))
    idfn = sum(ref_n.values()) - idtp
    idfp = sum(test_n.values()) - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    # fragmentation, weighted by frames: for each reference track, the
    # frame-weighted count of distinct test tracks that cover it.
    frag_num, frag_den, idsw = 0.0, 0, 0
    per_ref_frag = {}
    for a, seq in per_ref_seq.items():
        if not seq:
            continue
        distinct = len(set(seq))
        per_ref_frag[a] = distinct
        frag_num += distinct * len(seq)
        frag_den += len(seq)
        idsw += sum(1 for u, v in zip(seq, seq[1:]) if u != v)
    frag = frag_num / max(frag_den, 1)

    return dict(idf1=idf1, idtp=idtp, idfn=idfn, idfp=idfp, frag=frag,
                idsw=idsw, n_ref=len(R), n_test=len(T),
                per_ref_frag=per_ref_frag,
                ref_frames=sum(ref_n.values()), test_frames=sum(test_n.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("test")
    ap.add_argument("--gate", type=float, default=0.5,
                    help="matching gate as a multiple of local NN spacing")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    ref, _ = load(args.reference)
    test, _ = load(args.test)

    if args.sweep:
        print(f"  {'gate':>6}{'IDF1':>9}{'frag':>8}{'IDSW':>7}{'n_test':>8}")
        for g in [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]:
            m = evaluate(ref, test, g)
            print(f"  {g:>6.2f}{m['idf1']:>9.4f}{m['frag']:>8.3f}"
                  f"{m['idsw']:>7}{m['n_test']:>8}")
        return

    m = evaluate(ref, test, args.gate)
    if args.json:
        m.pop("per_ref_frag")
        print(json.dumps(m))
        return
    print(f"  ref {Path(args.reference).name}")
    print(f"  test {Path(args.test).name}")
    print(f"  gate {args.gate} x local NN spacing")
    print(f"  IDF1 {m['idf1']:.4f}   IDTP {m['idtp']}  IDFN {m['idfn']}  "
          f"IDFP {m['idfp']}")
    print(f"  frag {m['frag']:.3f}   IDSW {m['idsw']}   "
          f"ids ref {m['n_ref']} test {m['n_test']}")
    worst = sorted(m["per_ref_frag"].items(), key=lambda kv: -kv[1])[:8]
    print("  most-split reference tracks: " +
          ", ".join(f"{k}x{v}" for k, v in worst))


if __name__ == "__main__":
    main()
