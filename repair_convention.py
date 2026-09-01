"""Re-normalise a detection file that was converted under the wrong convention.

No API calls. The conversion detect.py applied is exactly invertible, so a run
mis-pinned as PIXEL can be recovered as THOUSANDTH arithmetically:

    saved   = raw / frame_dim          (what the wrong pin did)
    raw     = saved * frame_dim        (undo it)
    correct = raw / 1000               (what it should have been)

which collapses to  correct = saved * frame_dim / 1000.

Writes alongside the original with the corrected convention recorded, so the
mis-pinned file stays on disk as evidence rather than being overwritten.

    uv run repair_convention.py outputs/detections/foo.json --from pixel --to thousandth
"""

import argparse
import json
import shutil
from pathlib import Path

SCALES = {"pixel": None, "thousandth": (1000.0, 1000.0), "fraction": (1.0, 1.0)}


def factors(conv, w, h):
    if conv == "pixel":
        return float(w), float(h)
    return SCALES[conv]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--from", dest="src", required=True, choices=list(SCALES))
    ap.add_argument("--to", dest="dst", required=True, choices=list(SCALES))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    d = json.loads(args.path.read_text(encoding="utf-8"))
    if d.get("coord_convention") != args.src:
        print(f"  note: file records convention "
              f"'{d.get('coord_convention')}', you said '{args.src}'")

    sw, sh = factors(args.src, args.width, args.height)
    dw, dh = factors(args.dst, args.width, args.height)
    fx, fy = sw / dw, sh / dh
    print(f"  undo /{sw:g},/{sh:g}  then apply /{dw:g},/{dh:g}"
          f"   =>  x*{fx:.4f}  y*{fy:.4f}")

    backup = args.path.with_suffix(f".{args.src}.json")
    if not backup.exists():
        shutil.copy2(args.path, backup)
        print(f"  original kept at {backup.name}")

    n = 0
    for f in d["frames"]:
        for p in f["players"]:
            p["x"] *= fx; p["w"] *= fx
            p["y"] *= fy; p["h"] *= fy
            n += 1
        b = f.get("ball")
        if b:
            b["x"] *= fx; b["w"] *= fx
            b["y"] *= fy; b["h"] *= fy
    d["coord_convention"] = args.dst
    d["repaired_from"] = args.src
    args.path.write_text(json.dumps(d, indent=1), encoding="utf-8")

    ps = [q for f in d["frames"] for q in f["players"]]
    hw = sorted(q["h"] / max(q["w"], 1e-9) for q in ps)
    mx = max(max(q["x"] + q["w"], q["y"] + q["h"]) for q in ps)
    print(f"  rewrote {n} boxes")
    print(f"  h/w now med {hw[len(hw)//2]:.2f} (a standing player is 3-5)")
    print(f"  largest edge now {mx:.3f} (should be <= ~1.0)")


if __name__ == "__main__":
    main()
