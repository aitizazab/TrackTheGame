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

**2b · `bridge.py` / `bridge_live.py` — follow the pixels (built, not shipped)**
Sparse Lucas-Kanade optical flow through every source frame between two anchors,
a second pixel-measured estimate of where each player went, blended with the
Kalman prediction. The brief permits classical CV for *tracking* but never for
*detection*, and this sits exactly on that line: it never finds a player, only
follows one the model already found.

The **engineering** works and is worth reading: it runs concurrently with the
API calls rather than after them — decode depends on nothing and starts before
the first call, and each gap needs only its own anchor's detections — for
**0.11–0.14s of added wall clock against 5.9s run serially**, with output
verified byte-identical to the batch tool.

The **tracking claim did not survive validation.** It reduces basketball from 22
identities to 18, which looked like repaired fragmentation and is not: matching
markers by exact position shows all three merges are wrong, joining players who
are on court in the same frame. A lower identity count is what identity
*switching* produces too, and the count cannot tell the two apart. Not applied
to the deliverables. See D53 (retracted), D58.

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
| volleyball | $0.4373 | 23.9s | 14.9s / 17.6s | 147/150 |

**Mean $0.4662 per finished video**, stable to about ±$0.04 across five clips
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

### One run at 3fps

Sampling is 5fps because D9 measured that as the floor. Out of curiosity we ran
`allstars` — the crowded case, 22 players — at **3fps** to see what the cheaper
setting actually buys and costs. It was not planned as an ablation; it turned
into a real finding, so it is recorded here.

| | 5fps (shipped) | 3fps |
|---|---|---|
| cost | $0.5253 | **$0.3217** |
| wall clock | 26.6s | **14.4s** |
| frames returned | 147/150 | 90/90 |
| identities drawn | 29 | **32** |
| jersey numbers read | 16 | 14 |
| markers drawn | 15,450 | 15,010 |

**It is 39% cheaper and it is the only configuration we have ever run that meets
the original 15-second target.** It also tracks visibly worse.

The cost is identity fragmentation, and the labels name it precisely: at 3fps the
run *loses* real jersey numbers `30` and `93` and *gains* invented identifiers
`I`, `II`, `III` and `XVIII`. Three more identities for the same twenty-two
players means players are breaking into pieces.

The cause is the association gate, though not in the way this README first
claimed. The gate is capped at **1.5× the clip's own player spacing**, and that
cap binds on every clip at every sample rate — so the ratio is 1.50 by
construction, not the 3.6×/6.1× reported here earlier, which described an
uncapped code path the tracker never executes. The gate does not widen at 3fps;
it stays pinned while real motion grows 1.67×, so it is relatively *tighter* and
real matches fall outside it. Scaling the cap for the longer interval recovers
two of the three lost identities.

**Not shipped.** It produces a working annotated video in the literal sense, and
it is worse, so the deliverable stays at 5fps. The detections and tracks are
committed like every other run, so the render reproduces without an API key:

```bash
uv run render.py outputs/tracks/allstars_fr_eng_1080__allstars_3fps__tracks.json     --clip clips/allstars_fr_eng.mp4 --ball-fade --out allstars_3fps.mp4
```

### Going lower, measured live

4fps and 2.5fps were later run for real, and both are worse than 5fps on
every clip. The figures below are **live runs**, not decimations:

| | cost | wall | identities (roster) | ID switches |
|---|---|---|---|---|
| basketball 5fps | $0.4407 | 21.2s | 22 (10) | reference |
| basketball 4fps | $0.3400 | 27.3s | 22 (10) | 46 |
| basketball 2.5fps | $0.2095 | 26.2s | 28 (10) | 73 |

Cost falls as expected; **wall clock does not**, because it is set by the
slowest single call and there is a slow call at any sample rate.

A methodological warning worth more than the result. Lower rates were first
estimated by *decimating* a 5fps run — dropping every other anchor — which is
valid for isolating a tracker change but not for estimating quality, because
**every decimated frame is byte-identical to the reference it is scored
against** (75/75) while a live run shares none (0/74). Decimated basketball at
2.5fps scores 0.767 IDF1; the live run at the same rate scores **0.595**. The
harness flattered itself and we believed it for a day.

The pattern across every low-rate experiment: reduced sampling survives on
football-type broadcast footage and fails on a tight basketball camera. A
footballer at broadcast distance barely moves in image terms between anchors; a
basketball player covers several body heights and changes direction inside one.

So 5fps is not obviously the optimum in principle — it was chosen from a
geometric argument and confirmed rather than searched — but everything below it
has now been searched, and 5fps wins.

> **On latency.** Provider variance is larger than anything in our control, and
> it has now been measured twice. The same basketball clip, same configuration,
> ran **37.4s on one run and 21.2s on another** — a 43% swing with no code
> change. Volleyball, the slowest row above, was re-run under the identical
> configuration and came back at **23.9s against 36.5s**, a 34% swing in the
> same direction. Neither clip is expensive; both had a bad draw. **The
> volleyball row above is now that re-run**, which is the one that produced
> the committed video; the 36.5s original is kept here as the comparison.
> That is why the target is
> reported as a range rather than a figure. A dynamic straggler cut abandons the
> slowest 3% of calls once 97% have returned, which costs five frames of 150 and
> a worst blind spell of 0.40s, comfortably inside the tracker's 0.60s coast.
>
> **The cut looked useless for a while, and the diagnosis was one layer off.**
> Instrumenting the volleyball re-run showed the last useful call landing at
> 22.00s while the run did not end until 23.92s. The cause was not the cut but
> `with ThreadPoolExecutor(...)`, whose exit calls `shutdown(wait=True)` and
> blocks until every worker finishes — including the stragglers whose answers
> had just been abandoned. We stopped *using* the slow results and then waited
> for them anyway. Fixed by waiting with a deadline
> (`wait(..., timeout=..., return_when=FIRST_COMPLETED)`) and shutting down with
> `wait=False, cancel_futures=True`. Confirmed live: 16 calls with the cut
> forced at 50%, three abandoned, process out **0.33s** after the last kept call
> against 1.92s before.
>
> **The mechanism works; the saving does not.** Run end to end on
> `football_cuts` with the cut enabled: **22.3s against the shipped run's
> 22.4s**, on a worse provider draw. The 8–20% once projected here compared p97
> to the observed maximum and ignored the 1.5s grace period and 1.26s of
> teardown-and-write that sit between them. Kept enabled because it is correct,
> free, and bounds the tail on a bad day — but it is not a latency saving and
> should not be quoted as one.
>
> One further gap: **`ffmpeg` frame extraction (2.09s) sits outside the wall
> clock above**, which is measured across the call pool. True end to end for
> volleyball is about 26s.

| requirement | target | result |
|---|---|---|
| cost per video | under $1.00 | **$0.47** ✅ |
| processing time | under 15s, 25s accepted | 21–27s ⚠️ |

The processing-time row is the shipped runs. A later run with the repaired
straggler cut enabled came back at 22.3s against the same clip's 22.4s, so the
range above stands — provider variance (43% on identical configurations) is
still larger than anything in our control.
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

**Cost cannot go much below $0.47, and we know exactly why.** Between 60% and
78% of every output token is the model's own reasoning, which we pay for at the
same rate as the answer ($1.87/1M output, $0.38/1M input, fitted to five
decimals) and then discard. It is the only pool large enough to matter — the
answer itself is 33.6–37.0 tokens per player across three sports, near the floor
for four numbers, a colour word and a jersey number.

The API exposes two ways to shrink it and **both are the same switch**.
`effort: low` zeroes reasoning; `reasoning.max_tokens: 600` also zeroes it
rather than granting a budget. With reasoning off, two of three test frames
returned **every box off the 0..1 scale**. The answer length barely changes
(318 → 328 tokens) — only the coordinates break, because the model is not
reasoning about sport, it is doing per-player arithmetic to normalise pixel
positions into fractions. Reasoning here is load-bearing.

**Adaptive sampling from optical flow does not generalise.** Profiling motion
before any API call and densifying only the intervals that need it covers 57%
of the fragmentation on `allstars` against 25% expected by chance — but exactly
25% on basketball, which is chance. A tight camera on continuous play has no
quiet interval to steal frames from. The profile also costs 5.4s that cannot be
overlapped, since the schedule must be fixed before dispatch.

---

## Layout

```
clips/              five 30s inputs, 720p to render and 1080p to detect
outputs/detections/ per-frame model output — the evidence behind every claim
outputs/tracks/     tracker output; re-render from here with no API key
outputs/bridge/     optical-flow predictions, one file per run
outputs/videos/     the five deliverables (+ legacy/ for superseded renders)
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
