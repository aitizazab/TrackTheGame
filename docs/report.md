# Track the Game — method, frames, analysis

**Status: comprehensive draft.** The brief asks for three pages; this is
deliberately longer so that everything attempted is on the record and can be cut
down rather than remembered back. Sections marked ✂ are the first candidates for
removal if length binds.

---

## 1. The task and the constraints

Thirty seconds of game footage in, the same thirty seconds annotated out: every
player marked, the two teams marked differently, the ball highlighted, and the
player in possession marked differently again. FIFA-style markers under the feet.
Per the instructor's clarification, each player also carries their jersey number
above their head in their team's colour, and **that label must not mutate or
flash** — where no number is legible, an arbitrary but stable number is accepted.

| constraint | value | status |
|---|---|---|
| clip length | 30s, 900 frames @ 30fps | met |
| processing time | under 15s, 25s accepted | **31.0s — not met** |
| cost | under $1.00 per finished video | met, $0.8818 (→ ~$0.64 with `--compact`) |
| output | a video, not stills | met |
| deliverables | five clips, five videos | 4 clips collected, 1 video finished |

Two standing rules from the brief shaped everything: **VLMs for anything that
looks at the image, Pillow for anything that draws**, and **no agent frameworks**
— raw API calls only. A third came from the instructor mid-project: *"computer
vision methods can be used for object tracking purposes, not for detection."*

---

## 2. Architecture: two layers, and why the split is where it is

**The VLM sees each frame independently and reports what is in it. Classical
geometry links those reports across time.** Nothing that looks at pixels is
classical; nothing that maintains identity is a VLM.

That split is not a convenience. It is the only arrangement that satisfies the
instructor's clarification *and* the latency budget simultaneously — a tracker
operating over already-returned boxes costs milliseconds and adds zero round
trips, where any per-frame model call for identity would multiply them.

```
fetch_clips.py   download, cut to exactly 30.0s, force CFR 30fps, verify 900 frames
    ↓
detect.py        sample frames, one VLM call per frame, fully parallel, strict JSON schema
    ↓
track.py         Kalman + Hungarian association, identity, team assignment, ball filtering
    ↓
render.py        Pillow markers composited onto the source video
```

### Why clip normalisation is load-bearing ✂

Source footage is routinely variable-frame-rate. Extract frames by index from a
VFR file and the timestamps drift, so boxes returned for "frame 300" are drawn
onto a different instant than the model saw. The annotation slides out of sync
and it presents as a tracking bug. Forcing CFR 30fps once, up front, removes the
whole class of problem. One collected clip (`football_cuts`) is 23.976fps at
source, so this was not hypothetical.

---

## 3. The detection layer

### One frame per call, fully parallel

Not multi-frame windows. The concurrency probe showed median call time flat at
~3.9s all the way to N=64 with zero failures, so there is no meaningful
concurrency ceiling. Windows would only have amortised fixed per-call overhead,
which true parallelism pays once anyway, and would still have cost serial
generation inside each call.

### The schema, and why each field exists

Structured output via `json_schema`, strict, `additionalProperties: false`.

| field | why |
|---|---|
| `scene` | one sentence on framing and lighting, emitted **first** as a reasoning scratchpad before any number is committed |
| `x, y, w, h` | **bounding box, as fractions.** Box because the foot point comes out derived (`x + w/2`, `y + h`) and box height is a depth cue, letting the association gate scale with apparent size. Fractions because resolution is a planned ablation axis and pixel coordinates would need rescaling between runs, where any bug looks like a model difference |
| `kit` | the shirt colour **as an ordinary word**, never "team A" — each call is independent, so a call asked for "team A" picks its own A and the teams shuffle between frames. Colour is observer-independent |
| `num` | jersey number, **null unless genuinely readable**. A wrong number is far worse than no number |
| `role` | `outfield` / `goalkeeper`. Added after goalkeepers were silently excluded — see §7 |
| `conf` | model's own confidence, used to admit partly-hidden players rather than lose them |

### Coordinate conventions are pinned per model, never inferred

Models do not reliably obey the fraction convention the schema asks for. Gemini
3.x *flash-lite* returns 0–1000; 3.7-flash and Luna return fractions. A
three-bucket probe (`--probe-convention`) reports what a model actually did and
**refuses rather than guesses** when the answer is ambiguous; an unpinned model
will not run at all.

The tell that exposed this was in the data the whole time: `x+w` topped out at
1003 and 993 for two independent models — right at 1000, not near 1280 — while
`y+h` reached 835 and 865, which is impossible in a 720-pixel frame. **A ceiling
that lands on 1000 in both axes is a normalised space, not a resolution.** The
JSON looked perfectly valid; only the renders were wrong.

### `max_tokens = 4000`

Raised from 1600 after it failed on real frames. The model reasons before
emitting content and the cap governs both together. On three consecutive live
frames reasoning came in at 516, 1034 and 1306 tokens — a 2.5× swing — and at a
1600 cap two of the three truncated mid-string.

**Generalisable lesson, and it recurred:** a budget measured on an easy input is
not a budget. Both constants calibrated on the synthetic probe frame were wrong
by 3–5× against real footage.

---

## 4. The association layer

### Kalman + centre distance, not IoU

IoU-based association needs overlap and dies where displacement exceeds box
width — about 5–8fps here. Centre distance only needs the correct match to be
nearer than every wrong one, which holds while displacement stays under roughly
half the spacing between players. In a crowded box that spacing is 1–2m (19–38px
at our scale), giving a floor of 9–16fps for IoU against 2–3fps for centre
distance in open play.

### Labels are properties of the **track**, not the frame

Jersey number and team colour are decided **once per track**, by majority vote
over every observation in that track's lifetime, then painted onto every frame.

**This is what delivers the no-flicker requirement.** Not smoothing, not
thresholding — a label decided once has nothing to flicker between. It is only
possible because we are **offline**: every tracker in the literature is online
and cannot use the future. We can.

Fallback when no number is ever legible: the track's own ID, stable by
construction and rendered hollow so a viewer can tell bookkeeping from evidence.

### Two unit bugs, found by measuring rather than theorising

The first full run produced **358 identities for ~22 players** and reported 15
camera cuts on a clip containing none. Three explanations were proposed — camera
pan, same-team crossings, detector noise. **All three were wrong**, and one pass
of measurement settled it:

| quantity | measured |
|---|---|
| player movement between samples | p50 **0.0092** frac units |
| distance to nearest *wrong* candidate | p50 **0.072** |
| global camera shift per sample | p50 0.0065 (negligible) |
| association gate at the time | 0.059, covering **97%** of real moves |

The pan was negligible, the gate was adequate, and the correct match was
unambiguous 99% of the time. **The tracker was breaking on its own arithmetic.**

- **Bug 1 — OC-SORT retro-correction fired every frame.** Its threshold was
  written `gap > 2.5 / src_fps` — 2.5 *source* frames, 0.083s — while the normal
  gap between sampled frames at 10fps is 0.1s. Every ordinary update therefore
  overwrote the Kalman velocity with a raw two-point difference. The correct unit
  is the **sampling** interval, not the source one.
- **Bug 2 — measurement noise was 3× the signal.** `R` used `h * 0.35`, a
  standard deviation of ~0.029 against a median real movement of 0.0092. The
  filter believed its own drifting prediction over the detection in front of it.

| | before | after |
|---|---|---|
| raw tracks | 392 | **54** |
| identities | 358 | **49** |
| false cuts | 15 | **0** |
| match rate p50 | 0.80 | **0.905** |

Two further changes came from the same diagnostics: **kit disagreement became a
×4 penalty rather than a veto** (as a veto it killed ~1 track per frame — a
player read as "white" then "blue" through motion blur became unmatchable
despite unambiguous geometry), and the gate widened to 8 body-heights/sec.

### Ball precision, not ball recall

Recall is 97%. The failures are false positives: a pitch is covered in small
white round things — penalty spot, centre spot, painted arc, a boot, a sock — and
the model reports them confidently because they genuinely match the description.
Two purely geometric filters, rejecting 37 detections on the first clip:

- **Round trip.** If the ball leaps away and is back next sample where it
  started, the middle reading was a decoy. A real ball travelling that fast keeps
  going.
- **Over-speed.** Anything demanding a speed above 1.2 frac units/sec is a
  different object, not a fast ball.

---

## 5. Rendering

All drawing is Pillow; **the renderer reads no pixel it did not itself write.**

- **Flat ellipse under the feet** in the team's colour, at the box's bottom
  centre — which is why the schema asks for a box rather than a point.
- **Jersey number above the head**, same colour, black stroke so it reads against
  grass, crowd or kit. Invented IDs are drawn hollow, so a viewer can tell at a
  glance which labels are evidence and which are bookkeeping.
- **On the ball:** a soft glow on the player's own marker plus a brighter fill.
  An earlier hard white ring was removed — it was a second high-contrast edge
  competing with the ellipse it surrounded.
- **Crowding fade.** When two numbers overlap, *both* fade. Two numbers drawn on
  top of each other are worse than one: neither is readable and the frame looks
  broken. Fading turns an unresolvable collision into an honest signal.

### Team colour resolution

Each team uses its own kit colour. If the two kits are too close to tell apart,
team B switches to its **accent** colour; if that is also unusable, to the **hue
opposite** team A's, which is distinguishable by construction.

"Too close" is **CIE76 ΔE ≥ 30 in Lab space, not RGB distance.** RGB disagrees
with human vision badly enough to matter here: navy and black are far apart in
RGB and nearly identical on a floodlit pitch. *(Note: on all footage collected so
far the kits were distinguishable, so this fallback has never actually fired.)*

---

## 6. Everything tried

### Worked, and shipped

| change | evidence |
|---|---|
| **Provider pinning to the `flex` tier** | **−52% cost and −52% latency at identical quality.** See §8 |
| Native 1080p rather than 720p | latency flat against payload; jersey numbers 7.9% → 12.2% on the same clip |
| 5fps sampling | player spacing is constant (~0.050) while motion grows with `dt`; gate÷spacing goes 1.9× at 10fps → 3.6× at 5fps → 6.1× at 3fps. Failure appears between 3.6 and 6.1 |
| `--compact` (fixed-order arrays) | output tokens −28.9%, cost −26.9%, latency −12.2%, accuracy unchanged |
| Kit colour as a word | see §3 |
| Track-level label voting | the no-flicker requirement |
| Round-trip + over-speed ball filters | 37 decoys rejected on clip 1 |

### Tried and rejected

| change | why it failed |
|---|---|
| **`--ruler`** (coordinate reference drawn on the frame) | worse on both models — Gemini numbers 17.7% → 14.2%, Luna 21.5% → 20.1% — and reasoning went *up*. It solves "where is this"; our failure is "what does that shirt say" |
| **`--terse-schema`** | output tokens *up* 11%, cost up 4%, and it produced the only coordinate-corruption frame in its run. **The field descriptions were doing real work** |
| `--system` prompt on Gemini | +23% latency, no gain |
| `--scene-last` | helped Luna slightly, hurt Gemini — opposite signs on the same change |
| **Low reasoning effort** | 22% cheaper and it **destroys format compliance**: 36% of frames came back in a wrong coordinate scale, 13 mixing two scales inside one response. Reasoning effort controls instruction *adherence*, not just accuracy |
| **The cut detector** | 52 cuts across nine runs, **every one a false positive** on a corrupted-coordinate frame. Removing it took identities 49 → 32 |
| Camera-motion compensation for the ball filter | three variants, all worse. The camera estimate is not trustworthy enough to subtract |
| Auto-tuning the tracker constants | a single fixed number beat the whole adaptive system — the safety guards refuse to fire on exactly the runs that most need help |
| Shortening the coast / requiring re-confirmation | orphan markers 22 → 19 but missed detections 18 → 23. A wash |
| SportsMOT as a dataset | ships no frames without a competition signup, and its licence forbids redistribution — our clips go in a public repo. Its ground truth is positions and IDs only, so it would have scored one layer of the system, not the task |
| `--max-concurrent` capping | **arithmetically self-defeating** — see §8 |

### Models screened

| model | outcome |
|---|---|
| **`google/gemini-3.7-flash`** | **shipped.** Reliable, affordable on flex, adequate numbers |
| `openai/gpt-5.6-luna` | development baseline; best jersey numbers (12.2% on the 30s clip vs 9.0%), but 26.8s p50 and more identity swaps |
| `qwen/qwen3-vl-30b-a3b` | **fabricates.** 99.6% "read" jersey numbers, every box exactly 0.030×0.030, an invented squad list |
| `qwen/qwen3-vl-32b` | 28.3% read rate on paper, unusable on screen — invents markers, including an 11-ring "defensive line" that is a prior over football rather than a reading of the frame |
| `mistralai/mistral-large-2512` | **best read rate measured anywhere (22.2%) and still unusable.** Under-detects (12 players/frame vs 15), 8-word colour vocabulary including both `grey` and `gray`, match rate p50 **0.600** vs 0.933, 47 identities for ~22 players |
| `moonshotai/kimi-k2.5`, `z-ai/glm-5.3-flash` | no usable detections on the convention probe, twice each. Both have providers lacking `structured_outputs` |
| Frontier tier (Sonnet-5, GPT-5.1, Gemini 3.1 Pro, Grok 4.5) | priced out — at 150 calls the $1 cap is a ceiling of ~$3/Mtok output, and all are $6–25 |

---

## 7. What we got wrong

This section exists because most of the project's real findings are here. ✂ *(but
recommended to keep — it is the part that demonstrates method rather than result)*

### The metric problem, in both directions

**Six times a number said one thing and the video said another.** qwen ranked
first on read rate while unusable; 3fps "improved" identity inflation because
fewer samples means fewer chances to fragment; `--system` looked transformative
on a 50-frame worst-value statistic that halved at 146 frames; marker *count*
matched while *placement* did not; box aspect looked fine in fraction space
because a 16:9 frame inflates it by 1.78×; and Mistral won the read-rate metric
outright while failing at the association layer.

**Every metric here counts events; the artifacts are about placement and
continuity.** Use numbers to locate the moment worth watching, never to decide.

And it runs the other way too. On the finished clip, two defects were flagged
from the counts — 58 frames with no ball, and marker count falling to 13 against
a median of 18 — and **both were correct behaviour**: the ball was in the
goalkeeper's hands with the play dead, and there were genuinely only 13 players
in shot. **A flagged failure is a question, not a finding, until someone looks.**

### Goalkeepers were excluded by a single word

The schema said "one entry per **outfield** player". In football that term
specifically means "not the goalkeeper", so the model correctly followed an
instruction we did not intend. Goalkeepers are now explicitly required with a
`role` field, because a keeper in a third kit colour is otherwise
indistinguishable from an unstable colour word.

### A model comparison that varied four things at once

An early screen concluded that "every Gemini is near-blind to jersey numbers" and
that "newer is not better in either Gemini line". **Both were retracted.**

- The read-rate claim was a **resolution artifact**: every row was run at 1280 on
  the hardest clip in the project. The same model measured 0.2% there and
  **19.5%** at 1080p on ordinary footage — a ~100× spread with no prompt change.
- The "newer is worse" claim compared a *flash* against *flash-lites* (a tier
  difference reported as a version difference), rested on a model that **was
  never actually run**, and treated 0.8% vs 0.6% as a difference. On price,
  newer wins decisively: 3.7-flash is half the price of 3.5-flash.

**Two runs are comparable only when everything except the named variable is
pinned.** Tier, resolution, clip and provider all varied in that screen, and the
difference was attributed to whatever the heading named.

### A settled result overridden by a misremembered citation

`--compact` had been measured as "a free 23% saving" and written down. It was
then omitted from the signed-off run and justified by citing the
"tried and rejected" list — **which has never contained it**. That list rejects
`--terse-schema`, a different flag that moves the opposite way. One flag was
confused for the other, and the finished video cost 27% more than it needed to.
**Cite the line, not the memory of the line.**

### The budget ledger was incomplete in three ways

Per-call `usage.cost` was recorded from the API — the right instrument — and the
ledger was still wrong, because it was only ever as complete as the set of
*writers* someone remembered to check: a second project on the same key, a
schema that gained the cost field mid-project, and a probe script that never
recorded cost at all.

### ffmpeg's scene detector ranks football cuts backwards

Used to locate shot changes for the cut-handling clip. On real footage the three
genuine cuts scored **0.277, 0.306, 0.237** — *below* ordinary camera pan — while
title-card wipes in the same video scored **0.46–0.94**. Every football shot is
~70% green pitch plus crowd, so the frame histogram barely moves across a cut,
while a graphic replaces the entire palette. **No threshold fixes an inverted
ordering.** The cuts were found by inspecting a contact sheet instead. This is a
second, independent argument for associating on identity rather than pixels.

---

## 8. The ablation

The brief asks for parameters varied on purpose, one at a time, with a stated
reason. The axes below were chosen because each was *suspected to be the binding
constraint at the time it was run*.

### Completed

| # | varied | held | result |
|---|---|---|---|
| A1 | coordinate reference (none / ruler) | everything | ruler worse on both models; reasoning up |
| A2 | input resolution (640/960/1280/1920) | model, fps, prompt | **input tokens flat at 2821 across a 9× pixel range on Gemini** — resolution is free in *tokens*. Numbers climb 0.0 → 5.1% with width |
| A3 | model | prompt, fps, resolution | see §6 |
| A4 | `scene` field position | everything | opposite signs on the two models |
| A5 | reasoning effort | model, prompt | low effort destroys format compliance |
| A6 | sampling rate (1/2/3/5/10 fps) | everything | 5fps is the floor; 10fps buys nothing |
| A7 | schema verbosity | everything | terser schema costs *more* |
| A8 | **provider tier** | everything | **the largest single effect measured** |

### A8 — provider tier, the headline result

The same model is sold at three service tiers with identical context and identical
maximum output. Default routing silently moved between them mid-project, which had
been recorded in the project notes as a **price increase**. It was not; the cheap
tier was still live at 99.8% uptime.

Solving each logged call backwards from its own billed cost:

| run date | flex | standard |
|---|---|---|
| 28 Aug | **511 (100%)** | 0 |
| 31 Aug | 0 | **562 (100%)** |

Pinned A/B, same clip, same 100 frames, only the pin differing:

| | standard | **flex** |
|---|---|---|
| $ per call | 0.011109 | **0.005294** |
| latency p50 / p90 | 22.6 / 27.6s | **11.8 / 14.2s** |
| players/frame · numbers · box aspect | 15.0 · 19.1% · 0.102/0.026 | **identical within noise** |

**Half the cost and half the latency for a routing flag.** The quality columns are
indistinguishable — this is the same weights answering.

*Caveat kept deliberately: the latency halving is not cleanly attributable,
because pinning also sets `allow_fallbacks: false`, so the batch stops making a
routing decision per call. Tier and pinning changed together.*

### The remaining constraint, and the ablation it sets up

Wall clock is 31.0s against a 25s target. Diagnosis:

- Latency is **bimodal**, not a thin tail: 93 calls at p50 11.7s, 57 at p50 26.4s.
- The slow group is **not doing more work** — correlation of latency against
  output tokens is **+0.02**, and the slow calls emit *fewer* tokens.
- What predicts slowness is **position in the batch**: frames 30–74 run at
  11–13s with 0–7% slow; frames 120–149 are **100% slow**.

It is upload contention. All 150 frames upload at once, and **base64 inflates
48.4 MB of JPEG to 64.4 MB on the wire** — a 33% tax that the first analysis
missed entirely.

This rules out two obvious levers. **Lowering the timeout** does not help: it is
43s and never fires, so it cannot speed anything up, and at 25s it would discard
49 of 150 frames. **Capping concurrency** is worse than useless: the bytes must
cross the wire regardless, so capping only serialises them into waves — 3 waves
at the fast p50 is a ~35s floor, worse than the 31.0s we already have.

That leaves payload size, and **two independent ways to reduce it**:

| config | KB/frame | wire MB | projected wall |
|---|---|---|---|
| 1080p q90 (current) | 349 | 69.7 | 27.6s |
| **1080p q85** | 285 | 57.0 | **24.7s** |
| 1080p q80 | 249 | 49.8 | 23.1s |
| 720p q90 | 193 | 38.7 | 20.5s |

**A9, the final ablation: resolution versus JPEG quality at a matched byte
budget.** These fail differently and the outcome is genuinely uncertain.
Downscaling changes the *sampling grid* — an 8px jersey number becomes 5px and
the information is **gone**. Lowering quality keeps every pixel and coarsens the
rounding of high-frequency DCT coefficients — the number is **noisier but still
sampled**. You cannot un-average pixels, but you can often read through noise;
against that, a small digit *is* high-frequency content, which is exactly what
quantisation destroys first.

`quality=90` has been hardcoded since the first commit and never varied.

---

## 9. Results

Video 1 of 5, `gemini-3.7-flash` pinned to flex, 1080p, 5fps, 150 calls:

| | measured | target |
|---|---|---|
| **cost** | **$0.8818** (→ ~$0.64 with `--compact`) | under $1.00 ✅ |
| **wall clock** | **31.0s** | 25s accepted ❌ |
| frames returned | 150 / 150 | — |
| latency p50 / p90 / max | 13.9 / 27.3 / 30.3s | — |
| identities (~22 players) | 33 | — |
| **association match rate p50** | **1.00** | — |
| camera cuts / degenerate frames | 0 / 0 | — |
| ball drawn | 842 / 900 frames | — |
| jersey numbers read | 9.0% | — |

Detection reliability within an established track is **98%**, and the tracker
tolerates roughly 50% of detections going missing at 5fps, so recall is not the
binding axis. Zero identities are born after the opening 0.4s — labels are
nailed down for the clip's duration, which is the no-flicker requirement met.

---

## 10. Limitations, honestly

- **Jersey numbers read at 9–12%.** A player in a wide shot is ~70px tall, so the
  number on their back is 8–10px — genuinely below what the encoder resolves. The
  fallback path is therefore the *common* case, not the exception. The spec
  permits an arbitrary-but-stable number, and that is what most players get.
- **Wall clock misses the target** at 31.0s against 25s. Diagnosed, not fixed.
- **Cut handling is untested.** The pixel-based detector was removed as worse
  than useless; the designed replacement re-anchors identity by jersey number,
  which is exactly the signal that is only 9–12% available. A clip containing
  three verified cuts has now been collected and this is the next thing measured.
- **The ΔE colour fallback has never fired**, so the branch is unexercised.
- **Box aspect is ~2.2 where a standing player is ~3.5.** Whether the error is in
  the top edge, the bottom edge, or the width is under inspection; only the
  bottom edge would affect marker placement.
- **One clip still to collect** — a tight camera where players leave frame,
  stressing track birth and death, which nothing collected so far touches.

## 11. Budget

$25 allocation, $19.96 used, $5.04 remaining. Every call's real charge is
recorded in `docs/run_log.jsonl`; the reconciliation and its three gaps are in
`decisions.md` D21.

The most expensive lesson was not a model choice but an **infrastructure**
default: unpinned provider routing silently doubled the price of every call for
three days, and it was recorded as a price rise rather than investigated.
