"""Find the reference track whose identity the bridge fixes most, so it can be
looked at rather than trusted."""
import json
import sys
from collections import defaultdict

import eval_ids

ref, _ = eval_ids.load(sys.argv[1])
A, _ = eval_ids.load(sys.argv[2])      # without bridge
Bv, _ = eval_ids.load(sys.argv[3])     # with bridge
G = 0.5


def seq(test):
    per = defaultdict(list)
    for f in sorted(ref):
        for a, b in eval_ids.frame_match(ref[f], test.get(f, []), G):
            per[a].append((f, b))
    return per


sa, sb = seq(A), seq(Bv)
rows = []
for r in sa:
    na = len({b for _, b in sa[r]})
    nb = len({b for _, b in sb.get(r, [])})
    rows.append((na - nb, na, nb, len(sa[r]), r))
rows.sort(reverse=True)
for d, na, nb, n, r in rows[:6]:
    fr = [f for f, _ in sa[r]]
    print(f"  ref {r:<12} frames {fr[0]}-{fr[-1]} ({n})  "
          f"distinct test tracks: no-bridge {na} -> bridge {nb}")
    # where does the no-bridge run switch?
    prev = None
    for f, b in sa[r]:
        if prev is not None and b != prev:
            print(f"      no-bridge switches at frame {f}: {prev} -> {b}")
        prev = b
    prev = None
    for f, b in sb.get(r, []):
        if prev is not None and b != prev:
            print(f"      bridge    switches at frame {f}: {prev} -> {b}")
        prev = b
