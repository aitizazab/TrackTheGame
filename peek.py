"""Cut specific source frames and draw one named track's box, so a claimed
identity fix can be looked at instead of trusted.

Pillow draws; nothing here looks at the image.
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

clip, tracks_path, track_id, frames, out = (
    sys.argv[1], sys.argv[2], sys.argv[3],
    [int(x) for x in sys.argv[4].split(",")], Path(sys.argv[5]))
out.mkdir(parents=True, exist_ok=True)

T = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
by = {f["frame"]: {p["id"]: p for p in f["players"]} for f in T["frames"]}

tiles = []
for fr in frames:
    png = out / f"f{fr}.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip,
                    "-vf", f"select=eq(n\\,{fr})", "-vframes", "1", str(png)],
                   check=True)
    im = Image.open(png).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    p = by.get(fr, {}).get(track_id)
    if p:
        x, y, w, h = p["x"] * W, p["y"] * H, p["w"] * W, p["h"] * H
        d.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=4)
        d.text((x, max(0, y - 14)), f"{track_id} f{fr}", fill=(0, 255, 0))
    else:
        d.text((10, 10), f"{track_id} ABSENT f{fr}", fill=(255, 0, 0))
    # crop around the box so the player is legible in a contact sheet
    if p:
        cx, cy = x + w / 2, y + h / 2
        s = max(w, h) * 0.85
        box = (max(0, cx - s), max(0, cy - s), min(W, cx + s), min(H, cy + s))
        im = im.crop([int(v) for v in box]).resize((360, 360))
    else:
        im = im.resize((360, 360))
    tiles.append(im)

sheet = Image.new("RGB", (360 * len(tiles), 360))
for i, t in enumerate(tiles):
    sheet.paste(t, (360 * i, 0))
sheet.save(out / f"sheet_{track_id.replace('/', '_')}.png")
print(out / f"sheet_{track_id.replace('/', '_')}.png")
