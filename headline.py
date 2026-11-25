"""The final table: for each clip, base / best-parametric / bridge, with the
identity ceiling and the share of the identity headroom each recovers.

Ceiling = the IDF1 a run would score if its identity assignment were perfect,
i.e. every reference foot that has ANY test foot within the gate counts as a
true positive. It is set by what the run DETECTED, not by how it linked things,
so it is the honest upper bound on anything an identity mechanism can buy.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import eval_ids

TMP = Path("outputs/tracks/_sweep")
GATE = 0.5


def run(dets, flags, tag):
    out = Path("outputs/tracks") / (Path(dets).stem + "__tracks.json")
    backup = out.read_bytes() if out.exists() else None
    r = subprocess.run([sys.executable, "track.py", dets] + flags,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        sys.exit(tag)
    TMP.mkdir(parents=True, exist_ok=True)
    dst = TMP / f"hl_{tag}.json"
    shutil.copy(out, dst)
    if backup is not None:
        out.write_bytes(backup)
    return dst


def score(ref, test):
    m = eval_ids.evaluate(ref, test, GATE)
    tot = mat = 0
    for f in sorted(ref):
        tot += len(ref[f])
        mat += len(eval_ids.frame_match(ref[f], test.get(f, []), GATE))
    nt = sum(len(v) for v in test.values())
    ceil = 2 * mat / max(2 * mat + (tot - mat) + (nt - mat), 1)
    return m, ceil


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(f"  gate {GATE} x local NN spacing")
    print(f"  {'clip / config':<38}{'IDF1':>8}{'ceil':>8}{'frag':>7}"
          f"{'IDSW':>7}{'ids':>6}{'headroom recovered':>20}")
    for clip, refp, dets, variants in spec:
        ref, _ = eval_ids.load(refp)
        basev = None
        for tag, flags in variants:
            dst = run(dets, flags, f"{clip}_{tag}".replace(" ", "_"))
            test, _ = eval_ids.load(dst)
            m, ceil = score(ref, test)
            if basev is None:
                basev = m["idf1"]
                rec = ""
            else:
                rec = f"{100 * (m['idf1'] - basev) / max(ceil - basev, 1e-9):>17.0f}%"
            print(f"  {clip + ' / ' + tag:<38}{m['idf1']:>8.4f}{ceil:>8.4f}"
                  f"{m['frag']:>7.3f}{m['idsw']:>7}{m['n_test']:>6}{rec:>20}")
        print()


if __name__ == "__main__":
    main()
