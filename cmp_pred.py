"""Paired comparison of the two predictors on identical association decisions.

Each row is one (track, detection) pair the SHIPPED tracker accepted. `kal` is
how far the Kalman prediction landed from that detection, `brg` how far the
optical-flow bridge landed. Both aspect-corrected. `nn` is the median
nearest-neighbour player spacing in that frame, the unit in which a difference
starts to be able to mean "a different person".
"""
import json
import sys

import numpy as np

allrows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
rows = [r for r in allrows if not r.get("unmatched")]
un = [r for r in allrows if r.get("unmatched")]
k = np.array([r["kal"] for r in rows])
b = np.array([r["brg"] for r in rows])
nn = np.array([r["nn"] for r in rows])
h = np.array([r["h"] for r in rows])
ng = np.array([r["n_in_gate"] for r in rows])

print(f"  {len(rows)} paired decisions")
for name, unit, kk, bb in [("raw fraction", "", k, b),
                           ("/ NN spacing", "nn", k / nn, b / nn),
                           ("/ body height", "bh", k / h, b / h)]:
    print(f"  {name:<14} kalman p50 {np.median(kk):.3f} p90 "
          f"{np.percentile(kk, 90):.3f}   bridge p50 {np.median(bb):.3f} p90 "
          f"{np.percentile(bb, 90):.3f}")
print(f"  bridge closer than kalman: {100 * (b < k).mean():.1f}% of pairs")

print("\n  split by how many detections sit inside the gate "
      "(1 = unambiguous, >1 = the decision could go wrong)")
print(f"  {'n_in_gate':>10}{'pairs':>8}{'kal/nn p50':>12}{'brg/nn p50':>12}"
      f"{'brg wins':>10}{'kal>1nn':>9}{'brg>1nn':>9}")
for lo, hi, lab in [(0, 1, "<=1"), (2, 2, "2"), (3, 3, "3"), (4, 99, ">=4")]:
    m = (ng >= lo) & (ng <= hi)
    if m.sum() < 5:
        continue
    print(f"  {lab:>10}{int(m.sum()):>8}{np.median(k[m] / nn[m]):>12.3f}"
          f"{np.median(b[m] / nn[m]):>12.3f}"
          f"{100 * (b[m] < k[m]).mean():>9.1f}%"
          f"{100 * (k[m] / nn[m] > 1).mean():>8.1f}%"
          f"{100 * (b[m] / nn[m] > 1).mean():>8.1f}%")

print("\n  tail: a prediction further than one NN spacing from the taken "
      "detection is a prediction that could have pointed at someone else")
for q in [50, 75, 90, 95, 99]:
    print(f"    p{q:<3} kalman {np.percentile(k / nn, q):.3f}   "
          f"bridge {np.percentile(b / nn, q):.3f}")

if un:
    uk = np.array([r["kal_near"] for r in un])
    ub = np.array([r["brg_near"] for r in un])
    ug = np.array([r["gate"] for r in un])
    print(f"\n  UNMATCHED track-frames that had a bridge: {len(un)}")
    print(f"    nearest detection / gate: kalman p50 {np.median(uk / ug):.2f}  "
          f"bridge p50 {np.median(ub / ug):.2f}")
    print(f"    someone in gate: kalman {int((uk <= ug).sum())}  "
          f"bridge {int((ub <= ug).sum())}  "
          f"bridge-only rescues {int(((ub <= ug) & (uk > ug)).sum())}")
