"""Make a lower-rate detections file by dropping anchors from a higher-rate one.

WHY. Only two clips were ever run at 3fps, and detect.py costs real money, so
the bridge's effect could be measured on two clips or none. Dropping every
second anchor from a shipped 5fps run produces a genuine 2.5fps run on all five
clips at zero spend, and it is a STRICTER test of the same thing: the gap the
bridge has to span goes from 6 source frames to 12.

It is not identical to a real 2.5fps run - the model would have answered
slightly differently on a different frame set - but it is not a simulation
either. Every detection in it is a real model output on a real frame, and the
reference it is scored against is the run those detections came from, so the
detection content is a strict subset and any identity difference is the
tracker's, not the model's.
"""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
keep = int(sys.argv[2]) if len(sys.argv) > 2 else 2
d = json.loads(src.read_text(encoding="utf-8"))
frames = sorted(d["frames"], key=lambda f: f["frame"])
d["frames"] = frames[::keep]
d["fps"] = (d.get("fps") or 5) / keep
out = src.with_name(src.stem + f"_dec{keep}.json")
out.write_text(json.dumps(d), encoding="utf-8")
print(f"  {len(frames)} -> {len(d['frames'])} anchors, fps {d['fps']}  -> {out}")
