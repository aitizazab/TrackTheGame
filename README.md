# Track the Game

Thirty seconds of sports footage in, the same thirty seconds annotated out:
every player marked, the two teams marked differently, the ball highlighted, and
the player on the ball marked differently again.

A vision-language model does the *seeing*. Plain geometry does the *remembering*.
The model never learns that frames come in a sequence — it answers each one in
isolation, and nothing it returns carries identity. Everything that makes a
marker stay attached to a person happens afterwards, on the CPU, in
milliseconds.

**Five clips, three sports, five annotated videos.** Built in about ten days as
an internship project at Zeta Solutions.

---

## Watch these first

| video | what it is |
|---|---|
| [`outputs/videos/FINAL_allstars_fr_eng.mp4`](outputs/videos) | England v France. Broadcast wide, 22 players, the crowded case |
| [`outputs/videos/FINAL_basketball.mp4`](outputs/videos) | Tight camera, 10 players, large legible numbers |
| [`outputs/videos/FINAL_football_amateur.mp4`](outputs/videos) | Amateur match, auto-follow camera that pans hard, flat light |
| [`outputs/videos/FINAL_football_cuts.mp4`](outputs/videos) | Four hard camera cuts and a goalkeeper close-up |
| [`outputs/videos/FINAL_volleyball.mp4`](outputs/videos) | A third sport: a net, no goalkeepers, a ball almost always airborne |

Arabic numerals are jersey numbers actually read off a shirt. Roman numerals
(`VII`, `XII`) are invented but stable identifiers for players whose number
could not be read — which, at broadcast distance, is most of them.

---

## How it works

**1 · `fetch_clips.py` — normalise**
Pulls candidate footage at 1080p, cuts exactly 30.0 seconds, forces constant
30fps, and writes each clip twice: `<name>.mp4` at 1280×720 to render onto, and
`<name>_1080.mp4` at 1920×1080 to detect from. Every clip is exactly 900 frames,
so frame *N* is the same instant in all of them.

Source footage is routinely variable-frame-rate. Extract frames from a VFR file
by index and the timestamps drift, so boxes for "frame 300" get drawn onto a
different moment than the model saw — annotation slides out of sync and it looks
like a tracking bug. Forcing CFR once, up front, removes the whole class of
problem.

**2 · `detect.py` — perceive**
Samples 5fps (150 frames), sends every frame concurrently to
`google/gemini-3.7-flash` under a strict JSON schema, and gets back a list of
player boxes, kit colours, jersey numbers where legible, and the ball.

One frame per call, all at once. Concurrency is effectively free on latency —
wall clock is set by the slowest single call, not by the queue — so the whole
clip is answered in about the time one frame takes. Coordinates come back as
fractions of the image, which is what lets detection run at 1080p while
rendering runs at 720p.

**3 · `track.py` — remember**
The layer the task is actually about, and it makes zero network calls. A Kalman
filter predicts where each player should be; Hungarian assignment matches
predictions to detections using distance, kit colour, jersey number and apparent
box height; camera motion is estimated from the detections themselves and
subtracted before comparing. Identity is voted once per track rather than
per frame, so a number misread on one frame cannot rename anybody.

**4 · `render.py` — draw**
Paints all 900 source frames: team-coloured ground rings, a distinct treatment
for the player in possession, labels above each head, eased motion, fades on
exit, and label collision resolution that *moves* overlapping labels before it
fades them.

```bash
uv run detect.py clips/basketball_1080.mp4 --fps 5 --compact \
    --model google/gemini-3.7-flash --prompt-version v4 \
    --provider-order google-ai-studio/flex
uv run track.py  outputs/detections/basketball_1080__<tag>.json
uv run render.py outputs/tracks/basketball_1080__<tag>__tracks.json \
    --clip clips/basketball.mp4 --ball-fade
```

Detections and tracks are committed, so the last two steps reproduce **without
an API key**.

---

## The numbers

| clip | cost | wall clock | latency p50 / p90 | frames returned |
|---|---|---|---|---|
| football_cuts | $0.4727 | 22.4s | 12.3s / 15.5s | 150/150 |
| allstars | $0.5148 | 26.6s | 16.6s / 18.9s | 147/150 |
| basketball | $0.4407 | 21.2s | 11.5s / 15.1s | 149/150 |
| football_amateur | $0.4654 | 22.6s | 14.2s / 17.4s | 149/150 |
| volleyball | $0.4438 | 36.5s | 11.0s / 17.6s | 146/150 |

**Mean $0.4675 per finished video**, stable to about ±$0.04 across five clips
and three sports. Every figure is the run that produced the committed video,
over successful calls only.

Where a **typical** call's time goes (basketball, 150 calls, medians): 1.41s
encoding the frame, 9.28s to first byte, 2.12s streaming the answer back, and
effectively zero parsing. Those account for the call completely — measured
against the call's own duration the residual is 0.001s.

They do **not** add up to the 21.2s wall clock, and should not be expected to.
All 150 calls dispatch inside 0.66s and run concurrently, so the run ends when
the *slowest* one lands: the median call takes 12.9s and the slowest takes
21.2s, which is the wall clock to within ten milliseconds. The gap is
call-to-call variance in time-to-first-byte (9.3s → 14.3s) and streaming
(2.1s → 7.5s), not a missing stage.

The consequence worth stating: **wall clock is set by the tail, not the
median.** Halving a typical call would not finish the video any sooner. That is
why the latency lever here is the straggler cut, and why trimming the prompt cut
cost substantially while barely moving wall clock — speed and cost are separate
problems with separate levers.

> **On latency.** Provider variance is larger than anything in our control. The
> same basketball clip, same configuration, ran **37.4s on one run and 21.2s on
> another** — a 43% swing with no code change, and the reason the table above
> reports the shipped run rather than a best of several. That is why the target is
> reported as a range rather than a figure. A dynamic straggler cut abandons the
> slowest 3% of calls once 97% have returned, which costs four frames of 150 and
> a worst blind spell of 0.40s, comfortably inside the tracker's 0.60s coast.

| requirement | target | result |
|---|---|---|
| cost per video | under $1.00 | **$0.47** ✅ |
| processing time | under 15s, 25s accepted | 21–37s ⚠️ |
| output is a video | yes | ✅ |
| every player marked | yes | ✅ |
| teams marked differently | yes | ✅ |
| ball highlighted | yes | ✅ |
| player on the ball marked again | yes | ✅ |
| labels stable, never mutating | yes | ✅ |

---

## What does not work, and why

Written down because a project that only reports its wins is not reporting.

**Ball decoys.** The model sometimes returns a boot, a sock, an advertising
board or a painted mark instead of the ball — almost always at a moment the real
ball is occluded. Five separate approaches were measured and rejected:

- *appearance* — decoys sit **inside** the real ball's distribution on
  confidence, size ratio and box aspect
- *camera-compensated motion* — decoy residual 0.058 against a real-ball median
  of 0.053; no separation
- *candidate lists* — asked for up to three candidates, the model returns a mean
  of **0.94**, so there is nothing to arbitrate between
- *occlusion awareness* — asking the model to declare the ball hidden removed 2
  decoys and introduced 3, at 10–24% more cost
- *positional recurrence* — every labelled decoy appears **once**, or twice
  separated by seconds, so no clustering threshold can catch them

**One camera cut is undetectable.** The cut at 8.8s in `football_cuts` has 18
players either side, a median box height moving 0.075 → 0.080, an identical kit
distribution, and a scene description *more* similar than a typical non-cut
boundary. Nothing in the detection stream distinguishes it.

**Jersey numbers read 9–32%** depending on shot scale. At broadcast distance a
number is 8–10 pixels tall, which is below what the encoder can resolve. Most
players therefore carry a stable invented identifier instead — permitted by the
brief, and stated rather than hidden.

---

## Layout

```
clips/              five 30s inputs, 720p to render and 1080p to detect
outputs/detections/ per-frame model output — the evidence behind every claim
outputs/tracks/     tracker output; re-render from here with no API key
outputs/videos/     the five deliverables
docs/decisions.md   every decision with the measurement that justifies it
docs/report.md      method, findings, ablation
docs/run_log.jsonl  every API call ever made, with its real cost
assets/fonts/       the label typeface, shipped so renders reproduce anywhere
```

`docs/decisions.md` is the interesting one if you want to know *why* rather than
*what*. It includes the things that were tried and thrown away, with the numbers
that condemned them.

---

## Ground rules this was built under

No agent frameworks — raw API calls only, on the principle that you should not
hide the loop before you have seen the loop. A vision-language model for
anything that looks at the image; classical computer vision permitted for
*tracking* but never for *detection*. One API key with a hard budget cap and no
replacement.
