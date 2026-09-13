"""Run track.py under a set of flag combinations and score each against a
reference tracks.json with eval_ids.py. Analysis harness; not part of the pipeline.

track.py writes to a fixed path derived from the detections stem, so every
variant would overwrite the same file. Each run is copied aside before scoring.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import eval_ids

TMP = Path("outputs/tracks/_sweep")


def canonical(dets):
    return Path("outputs/tracks") / (Path(dets).stem + "__tracks.json")


def run_variant(dets, flags, tag):
    out = canonical(dets)
    cmd = [sys.executable, "track.py", str(dets)] + flags
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit(f"track.py failed for {tag}")
    TMP.mkdir(parents=True, exist_ok=True)
    dst = TMP / f"{tag}.json"
    shutil.copy(out, dst)
    ids = json.loads(dst.read_text(encoding="utf-8"))["identities"]
    nread = sum(1 for v in ids.values() if v["number_source"] == "read")
    return dst, len(ids), nread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--spec", required=True,
                    help="json list of [tag, [flags...]]")
    ap.add_argument("--gates", default="0.3,0.5,1.0")
    args = ap.parse_args()

    ref, _ = eval_ids.load(args.ref)
    gates = [float(g) for g in args.gates.split(",")]
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    # track.py always writes the same path; keep the committed file intact
    canon = sweep_canonical = run_variant.__globals__["canonical"](args.dets)
    backup = canon.read_bytes() if canon.exists() else None

    hdr = f"  {'variant':<26}{'ids':>5}{'read':>6}"
    for g in gates:
        hdr += f"{'IDF1@' + str(g):>11}{'frag':>8}"
    print(hdr)
    try:
        for tag, flags in spec:
            dst, nids, nread = run_variant(args.dets, flags, tag)
            test, _ = eval_ids.load(dst)
            line = f"  {tag:<26}{nids:>5}{nread:>6}"
            for g in gates:
                m = eval_ids.evaluate(ref, test, g)
                line += f"{m['idf1']:>11.4f}{m['frag']:>8.3f}"
            print(line)
    finally:
        if backup is not None:
            canon.write_bytes(backup)


if __name__ == "__main__":
    main()
