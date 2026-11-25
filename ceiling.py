"""The identity headroom: how much of the gap to the 5fps reference is
identity error, and how much is the test run simply not drawing a player there.

Upper bound: pretend identity assignment is perfect - every reference foot that
has ANY test foot inside the gate counts as a true positive. That is the best
IDF1 any identity mechanism could reach without changing what is detected.
Anything the bridge can buy lives between the measured IDF1 and this ceiling.
"""
import sys
import eval_ids

ref, _ = eval_ids.load(sys.argv[1])
test, _ = eval_ids.load(sys.argv[2])

print(f"  {'gate':>6}{'IDF1':>9}{'ceiling':>10}{'headroom':>10}{'det.agree':>11}")
for g in [0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
    m = eval_ids.evaluate(ref, test, g)
    tot = mat = 0
    for f in sorted(ref):
        r, t = ref[f], test.get(f, [])
        tot += len(r)
        mat += len(eval_ids.frame_match(r, t, g))
    nt = sum(len(v) for v in test.values())
    ceil = 2 * mat / max(2 * mat + (tot - mat) + (nt - mat), 1)
    print(f"  {g:>6.2f}{m['idf1']:>9.4f}{ceil:>10.4f}"
          f"{ceil - m['idf1']:>10.4f}{mat / tot:>11.3f}")
