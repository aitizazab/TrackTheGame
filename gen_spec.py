import json
import sys
from pathlib import Path

base = sys.argv[1]
out = sys.argv[2]
spec = [["base", []]]
for fb in ("1.0", "2.0", "5.0"):
    b = (f"outputs/bridge/{base}_dec2.json" if fb == "2.0"
         else f"outputs/bridge/{base}_dec2_fb{fb}.json")
    for a in ("0.3", "0.4", "0.5"):
        spec.append([f"fb{fb} a{a}",
                     ["--bridge", b, "--bridge-mode", "blend",
                      "--bridge-alpha", a]])
Path(out).write_text(json.dumps(spec), encoding="utf-8")
