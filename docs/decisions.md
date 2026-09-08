# Decisions — Track the Game

Current state, not a diary. When a decision changes, **edit the entry** and note
what superseded it. The chronological account of how we got here belongs in the
thinking cap, not in this file.

> Lesson carried over from Ball Detector: `findings.md` was a chronological log
> containing several `### Decision` headings, one per experiment. Reading it
> later, the *first* decision was mistaken for the final one (fraction mode, when
> bbox actually shipped). A running log and a current-state summary must be
> separate documents. This file is the current state.

---

## D1 · Two layers: VLM perceives, geometry associates

The VLM sees each frame independently and reports what is in it. A classical
tracker links those reports across time. Nothing that looks at pixels is
classical; nothing that maintains identity is a VLM.

**Why:** the instructor's clarification — *"computer vision methods can be used
for object tracking purposes, not for detection"* — states the split explicitly.
It is also the only arrangement that fits the latency budget: a tracker over
already-returned boxes costs milliseconds and adds zero round trips.

---

## D2 · One frame per call, fully parallel, deadline-capped

**Not** multi-frame windows.

**Evidence** (`docs/budget_probe.jsonl`, conc sweep): median call time stayed
flat at ~3.9s all the way to N=64 with zero failures, so there is no meaningful
concurrency ceiling — all calls really do run at once. Wall clock is set entirely
by one straggler per batch (`slowest ≈ wall` in every row, and wall clock was
*non-monotonic* in N: 24.9s at N=16 against 18.3s at N=32).

Windows would only have amortised fixed per-call overhead, and with true
parallelism that overhead is paid once regardless of call count. They would still
have cost serial generation inside each call.

**Therefore the timeout is the primary latency control**, and the only one that
binds. Frame count is nearly free; the deadline is not.

---

## D3 · Native resolution (1280×720), no downscale

**Evidence:** latency is flat against payload. 320px → 1.77s to first data;
1280px → 1.96s, across a 14× byte range. Token cost is linear in pixel area
(≈ `0.00119 × pixels + 19`, so quadratic in width) but a 1120-token image is a
fraction of a cent.

Downscaling costs the pixels a football occupies and buys nothing measurable.
`--width` exists only to sweep resolution as an ablation.

---

## D4 · `max_tokens = 6500`

**Was 1600, then 4000. Raised twice; the current value is 6500 — see the note
at the end of this entry. The heading said 4000 until 3 Sep.**

The output sweep showed `completion_tokens` equalling `max_tokens` exactly at
every level, with every token below 128 being a reasoning token — so Luna
reasons first and the cap governs reasoning + content together. That gave an
estimate of ~260 reasoning tokens, and 1600 looked generous.

It was measured on a one-line prompt over a synthetic image, and it did not
survive a real crowded frame. First live run, three frames:

| frame | completion | reasoning | left for content | result |
|---|---|---|---|---|
| 0 | 1600 (capped) | 1306 | 294 | truncated mid-string |
| 150 | 1600 (capped) | 1034 | 566 | truncated mid-string |
| 300 | 1315 | 516 | 799 | fine |

Reasoning varies 2.5× frame to frame. 4000 covers ~2000 reasoning plus ~25
players of content. It is a ceiling, not a reservation — frame 300 billed 1315.

**Generalisable lesson:** a budget measured on an easy input is not a budget.
Both of the constants that have bitten us so far were calibrated on the probe's
synthetic frame and were wrong by 3–5× on real footage.

`detect.py` now names both failure modes distinctly: `finish_reason == "length"`
reports `"truncated at max_tokens"` with the reasoning count, and empty content
reports `"empty content"`. Truncation previously surfaced as a
`JSONDecodeError` about an unterminated string, which reads like a schema bug and
sends you looking in the wrong place.

**Second raise, 4000 → 6500.** Across 300 real frames reasoning reached 2578
tokens at the top end and two frames truncated anyway. 6500 is a ceiling, not a
reservation.

**Do not nudge it again.** Reasoning is 77.9% of output tokens on the shipping
model and ~64% of the per-video bill (D25), so the cap is not the lever — the
number of judgement calls the prompt demands is, which is what D26 acts on.

⚠ **Three different failures all surface as a `JSONDecodeError`** and must not
be conflated — this happened on 3 Sep and produced a wrong entry in D25:

| symptom | what it is |
|---|---|
| `finish_reason == "length"`, high `reasoning_tokens` | genuine truncation. Raise the cap |
| `compl=0, reason=0`, HTTP 200 | the **zero-token 200**: the provider returned an empty completion. Already retryable; not a schema problem |
| valid `finish_reason`, non-zero tokens, bad JSON | an actual malformed response. This has not yet been observed on the shipping model |

Check `completion_tokens` and `finish_reason` **before** reading the parse error.

---

## D5 · Bounding boxes, expressed as fractions

Boxes, because: feet come out derived (`x + w/2`, `y + h`) rather than asked for;
box height is a depth cue, letting the association gate scale with apparent size
instead of using one global threshold; and it matches MOT convention.

Fractions, because resolution is a planned ablation axis and pixel coordinates
would need rescaling between runs, where any bug looks like a model difference.

**Correction to an earlier claim:** bbox — not fraction mode — was the shipped
format for Luna on Ball Detector (`walkthrough_brief.md:288`). Fraction was
qwen's. The two are not in conflict: box *and* fractions.

---

## D6 · Kit colour as a word, never "team A"

Each call is independent, so a call asked for "team A" picks its own A and the
teams shuffle between frames. Colour is observer-independent. The colour → team
mapping happens once, globally, in the tracker.

---

## D7 · Labels are properties of the track, not the frame

Jersey number and team colour are decided **once per track**, by majority vote
over every observation in that track's lifetime, then painted onto every frame
of it.

This is what delivers the no-flicker requirement. Not smoothing, not
thresholding — a label decided once has nothing to flicker between. It is only
possible because we are **offline**: every tracker in the literature is online
and cannot use the future. We can.

Fallback when no number is ever legible: the track's own ID, which is stable by
construction. Per the spec, an arbitrary-but-consistent number is acceptable.

---

## D8 · Camera cuts: detect, terminate, re-anchor by number

**The gap this closes:** at a shot change every track's position becomes
meaningless simultaneously. Naive association will match players across the cut
by proximity and produce markers that teleport and identities that swap wholesale
in a single frame.

### When it happens

In `track.py`, **at the top of each sampled frame's processing, before
association**. Not a post-process, not a pre-pass over the video.

Order of operations per frame:

1. Predict all active tracks forward to this frame.
2. **Cut check** ← here.
3. If a cut: finalise and retire every active track, empty the track set, and
   birth new tracks from this frame's detections. Skip association entirely for
   this frame.
4. If no cut: proceed to normal two-stage association.

### How it is detected

Two signals, either sufficient, both free:

- **Association collapse.** If fewer than ~30% of active tracks find a match
  inside the gate, and there were ≥4 active tracks, treat it as a cut. Derived
  from detections only — no pixel access.
- **`scene` discontinuity.** `detect.py` already asks the model for one sentence
  on camera framing and lighting, as a reasoning scratchpad. A sharp change in
  that description is a shot-change signal supplied by the VLM at zero extra
  cost. Secondary confirmation for the first signal.

**Measured 1 Sep — and it vindicates using neither pixels nor a threshold.**
ffmpeg's `scene` filter, the obvious off-the-shelf cut detector, **ranks football
cuts backwards.** On the collected `football_cuts` clip, which contains three
verified shot changes:

| event | scene score |
|---|---|
| wide → goalkeeper close-up (real cut) | **0.277** |
| keeper → behind-goal angle (real cut) | **0.306** |
| goalmouth → wide, match clock jumps 02:45 → 06:24 (real cut) | **0.237** |
| *title-card transitions in the same video's intro* | **0.462 – 0.941** |
| *ordinary camera pan, same clip* | 0.20 – 0.25 |

A real gameplay cut is **indistinguishable from a pan**, while a graphic wipe
scores 3–4× higher. The reason is structural: every football shot is ~70% green
pitch plus crowd, so the frame histogram barely moves across a cut, whereas a
title card replaces the entire palette. **Any pixel-difference cut detector will
fail on this footage in exactly this way**, and tuning the threshold cannot fix
an ordering that is inverted.

This is a second, independent argument for D8's design — identity association is
the signal, not pixels — and a caution for the report: *"we used scene detection
to find the cuts"* would have been wrong, and it was, until the clips were
checked frame by frame.

### How identity survives it

It cannot survive geometrically — that is what a cut means. It survives by
**jersey number**: when a new track accumulates a modal number matching a
retired track's modal number, it inherits that track's display ID and colour.

This is the third distinct job the number-reading layer has turned out to do —
re-ID through occlusion, continuity across dropped straggler frames, and now
identity across shot changes. It is load-bearing, not a nice-to-have.

### Scope

Deliverable clips are chosen from sections without mid-clip cuts. Detect-and-
reset exists so that a cut degrades to *"identities restart"* rather than
*"markers explode"*, and we measure what actually happens on one clip that does
contain a cut. Per the brief, that measurement is a finding.

---

## D9 · Sampling at 10–15 fps

**Derivation, not convention.** In a wide shot at 1280px across a ~68m pitch
(≈19 px/m), a 9 m/s sprint is ~170 px/s and a player's box is ~22px wide.

IoU-based association needs overlap, and dies where displacement exceeds box
width — about 5–8fps. We avoid that by using **centre distance, not IoU**, which
only needs the correct match to be nearer than every wrong one. That holds while
displacement stays under roughly half the spacing between players:

| situation | spacing | floor |
|---|---|---|
| open play | 5–10 m (95–190 px) | 2–3 fps |
| **crowded box** | **1–2 m (19–38 px)** | **9–16 fps** |

The crowded case binds — it is also where a viewer would notice a swapped label.
Above ~15fps the marginal gain is small. Frames are nearly free (D2), so there is
no reason to sit below 10.

---

## D10 · SportsMOT: dropped entirely

Considered as a clip source and as an evaluation baseline. Rejected on both.

The GitHub repo ships code and ground truth but **no frames** — those need a
Codalab competition signup. And the Terms state *"You agree not to distribute the
SportsMOT dataset without prior written permission"*, under CC BY-NC 4.0. Our
clips get committed to a public repo, so it is unusable for the deliverable.

Its ground truth is also positions and track IDs only — no team labels, no
jersey numbers — so it would have scored one layer of our system, not the task.

---

## D11 · Marker design

- **Flat ellipse under the feet**, in the team's colour. Drawn at the bottom
  centre of the box, which is why D5 asks for a box rather than a point.
- **Jersey number above the head**, same colour, with a black stroke so it reads
  against grass, crowd or kit.
- **Crowding fade.** When two numbers come within ~2.4 glyph widths, *both* fade,
  further the closer they get, down to 22% opacity. Two numbers drawn on top of
  each other are worse than one: neither is readable and the frame looks broken.
  Fading turns an unresolvable collision into an honest signal that two players
  overlap.
- **On the ball: marked differently again** — a white halo ring outside the team
  ellipse, plus a brighter fill. The team colour stays inside the ring, so the
  viewer still reads *which side* has possession, not merely that someone does.
- **Ball**: white ring with a yellow outer ring.

### Team colour resolution

1. Each team uses its own kit colour.
2. If the two kits are too close to tell apart, team B switches to its **accent**
   colour — the trim/shorts/number colour, asked for once per frame in the
   schema and voted across the clip.
3. If that accent is itself unusable, fall back to the **hue opposite** team A's
   colour, which is distinguishable by construction.

"Too close" is **CIE76 ΔE ≥ 30 in Lab space**, not RGB distance. RGB disagrees
with human vision badly enough to matter here — navy and black are far apart in
RGB and nearly identical on a floodlit pitch.

One extension to the spec as given: the accent is checked against team A's
*drawn colour* as well as against A's accent. A's drawn colour is what B's
marker actually sits beside on screen, so it is the comparison that decides
whether a viewer can separate them.

---

## D12 · Tracker calibration — two unit bugs, measured not guessed

First full run gave **358 identities for ~22 players** and 15 camera cuts on a
clip containing none. Three theories were offered (camera pan, same-team
crossings, detector noise). All three were wrong, and measuring settled it in one
pass:

| quantity | measured |
|---|---|
| player move between samples | p50 **0.0092**, p90 0.024 fraction units |
| distance to the nearest *wrong* candidate | p50 **0.072** |
| global camera shift per sample | p50 0.0065, max 0.051 — 1 frame in 293 above 0.03 |
| association gate at the time | 0.059, covering **97%** of real moves |

So the pan was negligible, the gate was adequate, and the correct match was
unambiguous 99% of the time. The tracker was breaking on its own arithmetic.

**Bug 1 — OC-SORT retro-correction fired every frame.** The threshold was written
`gap > 2.5 / src_fps` — 2.5 *source* frames, 0.083s — while the normal gap
between sampled frames at 10fps is 0.1s. Every ordinary update therefore
overwrote the Kalman velocity with a raw two-point difference, and `predict()`
projected that noise forward into the next gate. The correct unit is the
**sampling** interval, not the source one.

**Bug 2 — measurement noise was 3× the signal.** `R` used `h * 0.35`, giving a
standard deviation of ~0.029 fraction units against a median real movement of
0.0092. The filter believed its own drifting prediction over the detection in
front of it. Now `h * 0.06`, near actual detector jitter.

Two further changes, both from the diagnostics rather than from theory:

- **Kit disagreement is a penalty (×4), not a veto.** As a hard veto it killed
  ~1 track per frame — a player read as "white" then "blue" through motion blur
  became unmatchable despite unambiguous geometry.
- **Gate widened to 8 body-heights/sec.** The measurements show a wide corridor
  between covering real motion and admitting a rival; the gate was at the
  bottom of it.

| | before | after |
|---|---|---|
| raw tracks | 392 | **54** |
| identities | 358 | **49** |
| false cuts | 15 | **0** |
| match rate p50 | 0.80 | **0.905** |

`track.py` now records association health every frame — match rate, births, and
whether a leftover track failed on geometry or on the kit gate. That diagnostic
is what found both bugs and it stays in.

**Lesson, and it is the third time:** every constant that has bitten us was set
from physics on paper or from the synthetic fixture, and was wrong against real
footage — `MAX_TOKENS` twice, the cut thresholds, and both of these. Derive to
get the shape; measure to get the number.

---

## D13 · Ball precision, not ball recall

Recall is **97%** (289 of 297 frames). The failures are false positives: a pitch
is covered in small white round things — penalty spot, centre spot, painted arc,
a white boot, a sock. The model reports them confidently because they genuinely
match the description.

Two purely geometric filters in `track.py`, rejecting 37 detections on the first
clip:

- **Round trip (14).** If the ball leaps away and is back next sample where it
  started, the middle reading was a decoy. A real ball travelling that fast keeps
  going. This is the boot-mistaken-for-ball case and it shows in the video as a
  one-frame flicker.
- **Over-speed (23).** Anything demanding a speed above 1.2 fraction units/sec is
  a different object, not a fast ball. Catches sustained drift onto a static
  decoy, which the round-trip test cannot see.

Prompt side: the model is now told explicitly that painted markings, boots and
socks are not the ball, and to check the object sits *above* the grass rather
than on it.

---

## D14 · Goalkeepers were excluded by a single word

The schema said *"one entry per **outfield** player"*. In football that term
specifically means "not the goalkeeper", so the model correctly followed an
instruction we did not intend. Goalkeepers are now explicitly required, with a
`role` field, because a keeper in a third kit colour is otherwise
indistinguishable from an unstable colour word.

Consequence: only the two most-seen colours become teams (D6), so a keeper's kit
maps to `team = None` and renders neutral. Whether that is the right treatment is
open.

---

## D16 · Model screen on the hard 10s clip

Six models, 100 frames each at 10fps native, `hard10_allstars.mp4` (11–21s, the
hardest window by crowding and number legibility).

| model | returned | cost | lat p50 | **numbers** | ball | h/w |
|---|---|---|---|---|---|---|
| `qwen3-vl-32b-instruct` | 89/100 | $0.079 | 32.3s | **28.3%** | 87% | 1.25 |
| `qwen3-vl-30b-a3b` | 19/100 | $0.040 | 43.2s | *99.6% — fabricated* | 100% | 1.42 |
| `gemini-2.5-flash-lite` | 73/100 | $0.126 | 14.9s | 1.9% | 79% | 2.60 |
| `gemini-3.1-flash-lite` | **100/100** | ~$0.27 | **7.5s** | 0.8% | 96% | 4.62 |
| `gemini-3.5-flash-lite` | 99/100 | $0.535 | 8.7s | 0.6% | 98% | 6.27 |
| `gemini-3.7-flash` | 99/100 | $0.623 | 23.4s | 0.2% | 97% | 3.95 |
| *gpt-5.6-luna (30s ref)* | 292/300 | $0.955 | 31.3s | 7.9% | 93% | 4.50 |

### ~~Every Gemini is near-blind to jersey numbers~~ — RETRACTED 1 Sep

**This was a resolution artifact, not a property of Gemini.** Every row in the
table above was run at **1280 native on the hardest clip in the project**. The
same model, `gemini-3.7-flash`, measured later at 1080p:

| run | clip | width | numbers read |
|---|---|---|---|
| D16 screen | `hard10` (hardest window) | 1280 | **0.2%** |
| `flex_30s` | `allstars` 30s (the deliverable) | 1920 | **9.0%** |
| `flex_10fps` | `first10` (easy opening) | 1920 | **19.5%** |

That is a ~100× spread on one model with no prompt or schema change. §3 had
already measured Gemini's read rate climbing 0.0 → 5.1% with width; this table
should have been re-taken at that point and was not.

At 1080p `3.7-flash` reads numbers at **19.5%** against Luna's 21.5% — the same
band, not a 15× gap. The Qwen-VL comparison is void anyway (D17, and the whole
line is rejected).

**What survives:** at 1280 on a crowded frame, *nothing* reads jersey numbers.
That is a statement about **8 pixels of shirt**, not about a vendor. The
generalisable form is the one already in §3 — **read rate is a function of
resolution first and model second**, and any model comparison run at less than
native resolution measures the resolution.

### The coordinate pinning worked

`gemini-3.1-flash-lite` height/width went 2.62 raw → **4.62** normalised, which
is a correct standing-player aspect (3–5). `3.5-flash-lite` likewise. Both were
pinned to PIXEL from a 3-call probe; neither needed a second look.

### ~~Newer is not better in either Gemini line~~ — RETRACTED 1 Sep

**Wrong, and it should never have been written.** Three separate faults:

1. **It compared across tiers.** `3.7-flash` is a *flash*; everything it was
   measured against is a *flash-lite*. "Most expensive, 3× slower than the
   lites" is a tier difference being reported as a version difference.
2. **`gemini-3.5-flash` was never run.** It is not among the nine models this
   project has ever called. The claim about the flash line rests on no data at
   all. On price, newer wins decisively — **`3.7-flash` is exactly half the
   price of `3.5-flash`** (0.75/3.75 against 1.50/9.00), and `3.5-flash` has no
   flex tier, so per video it is $0.69 against $3.20 — **4.6× cheaper**.
3. **The number-read gap was noise.** 0.8% against 0.6% on ~1500 sightings, both
   indistinguishable from zero, at a resolution now known to suppress the metric
   entirely (see the retraction above).

The latency half is contaminated too: `3.7-flash` at "23.4s" was measured before
provider tiers were understood (D18). Pinned to flex at 1080p it runs **11.8s
p50**, faster than the flash-lites it was said to be 3× slower than.

**The corrected position.** Within a tier, newer has been better every time it
has been checked here:

| older | $/Mtok | newer | $/Mtok | |
|---|---|---|---|---|
| `gemini-3.5-flash` | 1.50 / 9.00 | **`gemini-3.7-flash`** | **0.75 / 3.75** | half price |
| `openai/gpt-5-mini` | 0.25 / 2.00 | **`openai/gpt-5.6-luna`** | **0.20 / 1.20** | cheaper, and 21.5% numbers |

The two counterexamples are both *lite* or *pro* lines drifting up
(`3.1-flash-lite` → `3.5-flash-lite`, `2.5-pro` → `3.1-pro`), which is a pricing
decision per line, not a capability trend. **Version numbers are not ordered
across tiers**: `3.5-flash` is an expensive outlier while `3.6-flash` and
`3.7-flash` are both 1/5 its price and newer.

**The method lesson, which is the same one as D17 and §5.** Every leg of this
claim was a comparison that held nothing constant — different tier, different
resolution, different clip, different provider — and the difference got
attributed to the one variable named in the heading. **Two runs are comparable
only when everything except the named variable is pinned**, which is now the
rule D18 states for providers and applies identically to model comparisons.

### ~~The trade, sharply~~ — DISSOLVED 1 Sep

The screen concluded that no model was both fast and able to read numbers, and
proposed a hybrid to bridge it. **There was no trade.** Both horns were artifacts:

- `qwen3-vl-32b`'s 28.3% is not a read rate worth having (D17, and the line is
  rejected outright — it invents markers).
- `gemini-3.1-flash-lite`'s "~0% numbers" was 1280 native. Read rate is set by
  resolution first.

`gemini-3.7-flash` at 1080p pinned to flex does both: **11.8s p50, 19.5%
numbers, 100% reliable, $0.69/video** (D18, D19). The hybrid was a solution to a
problem created by measuring three models at the wrong resolution on the wrong
clip. It is not needed and should not be built.

> **Read D16 as a historical record, not as current state.** Every row was run
> at 1280 on `hard10`, before provider tiers were understood. The current model
> decision is **D19**; the tier finding that invalidates its cost and latency
> columns is **D18**.

---

## D17 · Read-rate alone is a gameable metric

`qwen3-vl-30b-a3b` reported a **99.6%** jersey-number read rate. It is invented:

- **Every box is exactly 0.030 × 0.030.** A template, not a measurement — real
  boxes scale with depth.
- Numbers are **1–25 plus one 39**, near-uniformly distributed. A plausible
  squad list, not what is legible on a pitch.
- Player counts climb **21, 20, 23 … 32, 33, 38**. There are 22 on a pitch.
- **109 exact-duplicate positions**; flat confidence.
- Only 19/100 calls returned, because generating the fabricated roster is slow.

This is the failure mode predicted at the outset for constrained generation, in
its purest form: the schema demands a list of players with numbers, the model
cannot read any, so it fills the structure with plausible values. Well-formed,
schema-valid, entirely fictional.

**Consequence for the method:** number-read rate has been the headline quality
metric all week and it can be maxed by hallucinating. It needs a companion check.
The cheapest reliable one is **box-size variance** — a real detector's box
heights spread with depth (Luna 0.05–0.17), a fabricator's do not (0.030 flat).
Any future model comparison reports both.

---

## D18 · Pin the provider. Routing was an unmeasured variable all along

`detect.py` sent no `provider` field for the whole project, so every call took
OpenRouter's default routing. That is not a neutral default — it silently chose
**which price tier, which quantisation, and which feature set** answered us.

### Gemini sells the same model at three prices

Same weights, same 1 048 576 context, same 65 536 max output:

| tag | in $/Mtok | out $/Mtok | uptime 30m |
|---|---|---|---|
| `google-ai-studio/flex` | 0.38 | **1.88** | 99.8% |
| `google-ai-studio` | 0.75 | 3.75 | 99.9% |
| `google-ai-studio/priority` | 1.35 | 6.75 | 99.4% |

**The "price rise" recorded on 30 Aug never happened.** Solving each logged call
backwards from its own `usage.cost` over the 1073 calls at 1080p:

| run date | tags | flex | standard |
|---|---|---|---|
| 28 Aug | `f37_low`, `f37_med`, `f37_med_fix` | **511 (100%)** | 0 |
| 31 Aug | `ctl_*`, `fps*`, `ab_*`, `g_*` | 0 | **562 (100%)** |

A clean before/after, not a blend. The flex tier is still live at the old price.
Default routing simply stopped selecting it, and because nothing recorded *which
endpoint answered*, a routing change was written down as a price change. That is
the fourth time a number moved for a reason we had not measured.

### Pinning it: measured A/B, 100 frames, same clip, same everything

`first10_1080.mp4`, 10fps, 1080p, identical prompt and schema. Only the pin
differs (`--provider-order google-ai-studio/flex`, `allow_fallbacks: false`).

| | standard (`ctl_10fps`) | **flex (`flex_10fps`)** |
|---|---|---|
| $ / call | 0.011109 | **0.005294** |
| **$ / 30s video @5fps** | $1.666 | **$0.794** |
| latency p50 / p90 / max | 22.6 / 27.6 / 36.4s | **11.8 / 14.2 / 18.1s** |
| wall clock | 37.6s | **19.1s** |
| players / frame | 15.0 | 15.0 |
| number read | 19.1% | 19.5% |
| box h / w | 0.102 / 0.026 | 0.103 / 0.026 |
| kit vocabulary | white 51 · blue 46 · orange 2 · red 1 | **identical** |
| identities (~22 real) | 30 | 32 |
| match rate p50 | 0.938 | 0.933 |

**Half the cost, half the latency, quality inside noise.** The prediction going in
was that flex is deprioritised serving and would cost latency. That was wrong in
the safe direction, and the reason is not established: pinning also sets
`allow_fallbacks: false`, so the batch stops making a routing decision per call.
Tier and pinning changed together and have not been separated.

Two loose ends, neither blocking: input tokens fell 2714 → 1844 for a
byte-identical image and prompt, which looks like implicit cache credit landing
differently on a pinned endpoint; and the control ran 31 Aug against flex on
1 Sep, so day-to-day load is uncontrolled. The **cost** halving is exact and
structural; the **latency** halving is real but unexplained.

### The rule this sets

**Pin the provider on any run whose numbers will be compared to another run.**
An unpinned run measures a model *and* a routing lottery, and reports the sum as
if it were the model. `rec["provider"]` is now recorded on every call so the
question can never again need solving backwards from cost.

### It is not only price — it is also correctness

`structured: true` in the catalogue is an **OR across providers**, not a
guarantee. Checked against the endpoints API:

| model | providers | quantisation | lacking `structured_outputs` |
|---|---|---|---|
| `z-ai/glm-5.3-flash` | 21 | fp4, fp8, unknown | **8 of 21** |
| `moonshotai/kimi-k2.5` | 8 | **fp4, int4**, unknown | **3 of 8** |
| `mistralai/mistral-large-2512` | 3 | unknown | 0 |

Both models with no-JSON providers failed the convention probe twice and were
dropped. Not proven to be the cause — `--probe-convention` returns before the
log write, so the per-call errors were discarded — but it is the leading
hypothesis and the probe should be fixed before either is retried.

Note also that the cheapest endpoint for a big model is routinely a **4-bit
quantisation of it**. Any future "this model hallucinates" finding is not a
finding about the model until the provider was pinned.

### What flex did not fix: wall clock, and why the timeout is the wrong lever

The 30s deliverable run (`flex_30s`, 150 calls) came in at **31.0s wall** against
a 25s acceptance target. The obvious response is to tighten `TIMEOUT_S`. **The
measurement says no.**

Latency on that run is **bimodal**, not a long thin tail: 93 calls at p50 11.7s
and 57 calls at p50 26.4s. A 20s deadline therefore discards **57 of 150 frames
(38%)**, not a handful of stragglers. D2's "wall clock is one straggler" was
derived from the concurrency probe and does not describe this run.

The slow group is not doing more work. Correlation of latency against output
tokens is **+0.02**, and the slow calls emit *fewer* tokens (2583 vs 2767 p50).
What predicts slowness is position in the batch:

| frames submitted | lat p50 | share > 20s |
|---|---|---|
| 0–14 | 26.1s | 53% |
| 30–74 | 10.9–13.2s | 0–7% |
| 75–119 | 11.4–16.3s | 13–40% |
| **120–149** | **25.3–26.4s** | **100%** |

It is **upload contention**, and this project has measured it before. From
`detect.py:719`: at 300 concurrent 1080p frames, 85 of 90 failures were
`TimeoutError('The write operation timed out')` mid-send, and *"the same code at
720p had ZERO transport failures, which is the control."* At 150 concurrent the
same constraint appears as latency rather than as failure. `t0` is set at
`detect.py:703` — after `encode()`, before the POST — so upload time is inside
`latency_s`, and CPU/GIL contention during encoding is *not*.

**Capping concurrency is the wrong fix.** The payload is 325 KB per frame:
**48.8 MB has to cross the wire regardless of how it is scheduled.** The 14.7s
fast/slow gap over 48.8 MB implies an effective uplink of ~25 Mbps, so roughly
15s of the 31s wall is pure upload. Capping only serialises that into waves —
`--max-concurrent 64` gives 3 waves at the fast p50, a ~35s floor, **worse than
the 31.0s we already have**. Only 96 is even arguably neutral.

**The lever is payload size:**

| width | KB/frame | MB per 150-frame video | wire seconds @25 Mbps |
|---|---|---|---|
| 1920 (current) | 325 | 48.8 | **15.3** |
| 1280 | 169 | 25.3 | **7.9** |
| 960 | 117 | 17.6 | 5.5 |
| 640 | 59 | 8.8 | 2.8 |

**This reopens D3 / §3, which is not the same as contradicting it.** *"Resolution
is free on Gemini — flat 2821 input tokens at 640, 960, 1280 and 1920"* is still
true, and it was decisive when **cost** was the binding constraint. Resolution is
free in *tokens* and expensive in *seconds*. Now that cost is solved (D18/D19)
and wall clock is the only failing target, the same measurement supports the
opposite decision. **A conclusion is only valid against the constraint that was
binding when it was taken.**

The trade at 1280 is ~7.4s of wall clock against jersey-number read rate, which
§3 measured climbing with width. That is the next ablation, and it is one run.

**Fifth instance of the same lesson:** a constant derived from the synthetic
concurrency probe (`TIMEOUT_S = 43.0`, and the straggler model behind it) did not
survive contact with the real batch. *(`TIMEOUT_S` is **35.0** now, lowered 2 Sep.
It does bind on slower clips — the cuts non-compact run lost 19 frames to it.)* Derive to get the shape; measure to get the
number.

---

## D19 · The shipping configuration — signed off 1 Sep, showcase video 1 of 5

**Frozen as the baseline.** Later iterations are measured *against* this, not
instead of it. Nothing below is provisional.

```
uv run detect.py clips/allstars_fr_eng_1080.mp4 --fps 5 \
    --model google/gemini-3.7-flash --tag flex_30s \
    --provider-order google-ai-studio/flex
uv run track.py  outputs/detections/allstars_fr_eng_1080__flex_30s.json
uv run render.py outputs/tracks/allstars_fr_eng_1080__flex_30s__tracks.json \
    --clip clips/allstars_fr_eng.mp4
```

| axis | value | why, in one line |
|---|---|---|
| model | `google/gemini-3.7-flash` | D16 screen; the only model that is both reliable and affordable |
| provider | **pinned, `google-ai-studio/flex`** | D18 — half the cost and half the latency of standard, quality inside noise |
| resolution | 1080p native | D3 / §3: latency flat against payload, Gemini bills a flat 2790 input tokens at every width |
| sampling | 5fps → 150 calls | D9 / §3: 3fps fails visibly, 10fps buys nothing over 5 |
| coordinates | fractions, `FRACTION` pinned | D5, and the convention table is measured per model, never inferred |
| `max_tokens` | 4000 | D4 |
| render target | 720p clip | fractions are resolution-independent, so the output stays ~14MB |
| variants | **none** — no `--compact`, `--system`, `--terse-schema` | ⚠ see below: omitting `--compact` was an ERROR, not a decision |

**Measured:** $0.8818 · 31.0s wall · 150/150 frames · 33 identities · match rate
p50 1.00 · 0 cuts · 0 degenerate frames · ball drawn 842/900.

Artifacts, all confirmed committable (`git check-ignore` clears each):

```
outputs/videos/allstars_fr_eng_1080__flex_30s.mp4        the deliverable
outputs/tracks/allstars_fr_eng_1080__flex_30s__tracks.json
outputs/detections/allstars_fr_eng_1080__flex_30s.json   evidence for the report
```

Do not let this render be swept into `outputs/videos/legacy/`.

### ⚠ This baseline is missing `--compact`, and that was a mistake

The 31 Aug session record already settled it: ***"`--compact` is a free 23%
saving"*** — output 2530 → 1798, no visible defect. It was then left out of the
signed-off run and justified by a citation to §4 *"tried and rejected"*, **which
has never listed `--compact`**. §4 rejects `--terse-schema`, a different flag
that goes the other way (output *up* 11%). One flag was confused for the other.

Re-measured 1 Sep, same clip, same 50 frames, only the flag differing:

| | control | `--compact` | |
|---|---|---|---|
| output tokens | 2530 | 1798 | **−28.9%** |
| content tokens (excl. reasoning) | 1330 | 796 | **−40.1%** |
| input tokens | 2714 | 2245 | −17.3% |
| **$ per call** | 0.011521 | 0.008424 | **−26.9%** |
| **latency p50** | 22.3s | 19.6s | **−12.2%** |
| players/frame · numbers · ball | 15.0 · 17.7% · 48/50 | 15.0 · 17.6% · 50/50 | unchanged or better |

**Cost:** the signed-off video would have been **$0.6448 instead of $0.8818**.
**Latency:** −12% is not incidental — wall clock is the only failing constraint.

The next baseline run takes `--compact`. The lesson is the reverse of the usual
one here: this time the *measurement was right and sitting in the document*, and
it was overridden by a misremembered citation. **Cite the line, not the memory
of the line.**

### The two flagged defects were both false alarms

Handed over with timestamps and predicted failures, per the §5 rule. The user
watched both and both dissolved:

| flagged from the numbers | what was actually happening |
|---|---|
| ball absent for 58 frames (t+27.83–29.37s, t+29.63–29.97s) | ball **in the goalkeeper's hands**, play dead. Correct |
| markers fall to 13 at t+3.20–3.37s against a median of 18 | there were **only 13 players in shot**. Correct |

Every prior instance of the metric problem was a false *negative* — a number
looked good and the video was bad. This is the mirror image, and it sharpens the
rule rather than weakening it: the counts did their only job, which is to point
at a moment. **A flagged failure is a question, not a finding, until someone
watches it.** Both would otherwise have been written into the report as known
defects of the system, which they are not.

### What was still open on this configuration

> **Updated 6 Sep.** Cost closed: $0.8818 here became a **$0.4662 mean across
> five clips**, via the flex tier (D18), the v1→v2 prompt rewrite (D29) and
> `--compact` (D25). Latency did not close — see below, and D38 for where a
> call's time actually goes.

**Wall clock, 31.0s against a 25s acceptance target** — the only unmet hard
constraint. The diagnosis is in D18: bimodal latency driven by upload contention
at the batch tail, so the lever is `--max-concurrent`, **not** `TIMEOUT_S`.

> **Partly superseded by D38.** The upload-contention story was built on a
> `latency_s` that **did not include frame encoding** — 1.41s of local CPU per
> call, invisible to a stopwatch around the HTTP request. Thread queue delay,
> the mechanism blamed here, measures **0.30s**. Shipping range is now 21–37s,
> and the dominant term is provider variance: **37.4s and 21.2s on the same
> clip, same configuration, one re-run apart.** A straggler cut at 97% caps the
> tail. Latency remains the one unmet constraint and is reported as a range.

Jersey numbers read at 9.0% on this clip against 19.5% on the easy opening 10s.
Per D7 that is survivable — a label is decided once per track by majority vote,
so a track needs one legible sighting in its lifetime, not one per frame — and
33 identities against ~22 players with zero mid-clip births says identity is
holding. It remains the weakest axis and the honest limitation for the report.

---

## D20 · Payload bytes, not resolution, and the base64 tax

> ## ⛔ RETRACTED 2 Sep — the bandwidth model below is not supported
>
> D20 claimed the `p50 → wall` gap is upload contention, and that cutting bytes
> (JPEG quality, resolution) would cut wall clock. **The data refutes it.**
>
> **The decisive test — same frame count, different resolution:**
>
> | run | n | res | MB on wire | p50 | **wall** |
> |---|---|---|---|---|---|
> | `google_gemini_3_7_flash` | 100 | 1280 | **22.4** | 23.3s | **43.9s** |
> | `ctl_10fps` | 100 | 1920 | **43.9** | 22.6s | **37.6s** |
> | `flex_10fps` | 100 | 1920 | **43.9** | 11.8s | **19.1s** |
>
> **Half the bytes produced a LONGER wall.** And eight runs at an identical
> 22.0 MB show walls from 30.7s to 43.6s — a 13s spread with payload held fixed.
> Correlation of the gap against bytes is **+0.43**, against request count
> **+0.44** — indistinguishable, and both too weak to be a mechanism.
>
> **What is actually true: wall clock IS the slowest call.** Measured, three
> runs: wall minus slowest call is 0.7s / 1.0s / 1.2s, which is the ffmpeg
> extract. D2 said this originally and was right; D20 talked itself out of it.
>
> **And cutting frames barely helps**, because the slow band is not a thin tail.
> Resampling `flex_30s`'s own 150 latencies:
>
> | frames sent | median worst-case call |
> |---|---|
> | 150 | 30.3s |
> | 100 | 30.3s |
> | 75 | 29.1s |
> | 50 | 29.1s |
>
> p75 is already 25.5s, so **a quarter of all calls sit in the 25–30s band**.
> Draw 50 or 150 samples and you hit ~30s either way. Fewer frames buys cost,
> not wall clock.
>
> **The one direct lever is the deadline, and it has a cliff:**
>
> | `TIMEOUT_S` | frames lost | wall becomes |
> |---|---|---|
> | 43s / 35s | 0 | 30.3s |
> | 30s | 1 (0.7%) | 29.1s |
> | **28s** | **5 (3.3%)** | **27.9s** |
> | 25s | **49 (32.7%)** | 24.9s |
>
> **The dominant variable is provider-side latency variance, which we do not
> control.** `flex_10fps` and `ctl_10fps` are the same 100 frames at the same
> resolution on the same model, and differ 2× in wall clock. That is the whole
> effect, and it is why a peer can report 11s on the same configuration.
>
> **Consequences:** the JPEG-quality ablation (A9/E2) loses its rationale and is
> **cancelled** — it was predicated on bytes driving the wall. Dropping to 720p
> will not fix latency either; it remains only a jersey-number trade. The base64
> observation stands as a fact (33% wire overhead) but explains nothing here.
>
> **The methodological failure is the familiar one, one level up.** The
> byte-vs-wall story was built on a *single run's* internal correlation — the
> fast/slow split within `flex_30s` — and never checked against runs that varied
> payload independently. One run cannot separate two variables that move
> together inside it.

**Superseded. Retained below as written, for the record.**

Wall clock is the only failing constraint (31.0s against 25s). D18 established
the mechanism as upload contention. The refinement: **what is actually on the
wire is 33% larger than the JPEG**, because every image is base64'd into a data
URL. 48.4 MB of JPEG is **64.4 MB sent**, which puts the effective uplink at
**~35 Mbps**, not the 25 first estimated.

`detect.py:405` hardcodes `quality=90`. It has never been varied, is not in the
ablation list, and is the only byte lever that does not touch resolution.
Measured offline on 5 real frames from the deliverable clip — **zero API calls**:

| config | KB/frame | wire MB | wire s | projected wall |
|---|---|---|---|---|
| **1080p q90 (current)** | 349 | 69.7 | 15.9 | **27.6s** |
| **1080p q85** | 285 | 57.0 | 13.0 | **24.7s** |
| 1080p q80 | 249 | 49.8 | 11.4 | 23.1s |
| 720p q90 | 193 | 38.7 | 8.8 | 20.5s |

### The ablation this sets up, and why it is better than `--width`

Two ways to spend the same byte budget, which should fail *differently*:

- **720p @ q90** (193 KB) — halves the pixel count, so an 8px jersey number
  becomes 4px. The information is **gone**.
- **1080p @ q80** (249 KB) — every pixel survives, with compression ringing. The
  number is **noisier but still present**.

**Hypothesis: at a fixed payload, lowering quality beats lowering resolution for
small-text legibility.** Honestly uncertain — JPEG destroys exactly the
high-frequency detail a small number *is*, so it could go the other way. That is
what makes it worth one run rather than an assumption.

This reframes A2. Resolution was ablated for **cost** (free on Gemini — flat 2821
input tokens at every width) and for **accuracy** (numbers climb with width). It
was never ablated for **wall clock**, which is now the binding axis, and quality
was never ablated at all.

### Why "send more frames, cut the deadline harder" does not work

Proposed as a latency fix: oversample, then let the tight deadline drop the
stragglers. It fails on this pipeline for a structural reason.

**Latency is not per-call and independent — the batch shares one uplink.** The
slow group is the *tail of an upload queue*, so adding frames lengthens the
queue and increases the share that misses any given deadline. At 300 frames @
q90 that is 139 MB ≈ 32s of pure upload, and a 25s cutoff would kill over half.
It also costs 300 calls ≈ $1.26 with `--compact`, over the cap.

The same coupling explains why the two obvious levers fight each other: dropping
slow calls frees bandwidth for their siblings, and adding calls steals it.
**Fix the bytes first; then oversampling becomes available.**

`TIMEOUT_S` is separately a non-lever: at 43.0 it **never fires** (max observed
30.3s), so lowering it cannot speed anything up, and at 25s it would discard 49
of 150 frames — 33% — to buy ~6s.

---

## D21 · The budget ledger was incomplete in three ways

> **Figures superseded 3 Sep.** The instructor raised the limit to **$35** and
> topped up the shared pool; remaining is now **~$9.77**. The reconciliation
> below is kept because the *method* is the point — the three structural holes
> it found are still the reason to query `run_log.jsonl` rather than estimate.

`/api/v1/auth/key` reported **limit $25, limit_remaining $5.04** → **$19.96 used**.
The project ledger totalled **$18.81**, a 6% gap consistent with the estimated
rows below.

| source | amount | why it was missing |
|---|---|---|
| TrackTheGame recorded `cost_usd` | $17.05 | — |
| TrackTheGame, 303 successful uncosted calls | ~$0.87 | `cost_usd` was **added to the record mid-project**; 297 `luna_native` calls predate it |
| Budget probes | ~$0.02 | `probe_budget.py` **has no cost field at all** |
| **BallDetector** | **$0.87** | **a second project on the same key**, its own log, cost nested under `usage.cost` rather than a flat field |
| **total** | **$18.81** | |

**The account is the instructor's, shared across the cohort.** Account-level
`total_credits ≈ total_usage ≈ $1445` — the pool is exhausted, which is not our
spend. Until it is topped up nothing runs, regardless of the $5.04 allocation.
Symptoms: HTTP **402** *"would exceed your available credits given your current
in-flight requests"* and **429** upstream rate limits.

**402 is a reservation, not a charge.** OpenRouter holds the maximum a request
*could* cost while it is open, so 150-way concurrency reserves far more than it
spends. This makes `max_tokens` (4000, against a `--compact` median output of
1798) and concurrency into *budget* levers, which they were not before.

**$5.04 finishes the project, but only with `--compact`:** 4 clips $2.56 +
rebuild clip 1 $0.64 + one ablation $0.64 = **$3.84**. Without it, $5.28 — over.

**The lesson.** `usage.cost` per call was the right instrument and the ledger was
still wrong, because it was only ever as complete as the set of *writers* someone
remembered to check. Same shape as D18: the number was right, the **scope** of
the number was wrong.

---

## D22 · The Hungarian solver could not decline a match

`linear_sum_assignment` on a SQUARE matrix must return a PERFECT matching. With
10 tracks and 10 detections every track took a detection whether one fitted or
not: "this track coasts" and "this detection is a new player" were not in the
solution space. Out-of-gate cells held `1e6`, which is **finite**, so a 4x kit
mismatch at cost 0.41 was a bargain beside it.

Basketball frame 108, from the cost-matrix dump:

| candidate | dist | kit x | cost |
|---|---|---|---|
| blue det5 | 0.0370 | 1.0 | **0.0373** |
| white det9 | 0.0836 | **4.0** | **0.4061** |

Track 9 took the white one — 2.3x farther, 10.9x dearer — because det9 was in
gate for **no other track**, so the solver's only alternative for that column was
1e6. It then redistributed the rest and pushed a white track onto a blue
detection too. Across frames 0-300, **43 of 483 accepted pairs (8.9%) crossed a
team boundary**.

**Neither knob could have fixed it.** Raising `KIT_MISMATCH_PENALTY` needed ~1e7
to beat the sentinel, at which point it is a hard veto and D12's
one-death-per-frame returns. The `cost < 1e5` post-filter only rejects a pair
whose OWN cell is the sentinel, never one forced by a sentinel elsewhere.

Fixed by padding the matrix: a coast column per track and a birth row per
detection, priced at `NO_MATCH_GATES = 2.5` gate widths. Swept — below 2.5 the
swap is also fixed but basketball loses two identities; above it the bug returns
unchanged, and 2.5 and 50.0 behave identically.

Measured: the track labelled `3` is on a non-white detection **0 frames of 22**
against 13 of 19 before. Identity counts and match rates unchanged on all four
clips. **This was never a kit-penalty tuning problem**, which is what D7's note
had implied for two days.

---

## D23 · Cut detection rebuilt on shot scale

Both signals D8 specified are measured unusable. Association collapse was
removed 28 Aug after 52 false positives. The scene-sentence fallback is no
better: word overlap between consecutive `scene` sentences is **0.26 at a cut
against 0.36 away from one**, and a 30s no-cut control spans the same 0.09-0.44
range throughout. The model rewrites its sentence every frame regardless.

**Shot scale is the signal.** A cut moves the camera so apparent player size
changes violently; a pan does not.

| event | d median box height | d player count |
|---|---|---|
| t+17.2s cut | **6.64** | 8 |
| t+21.6s cut | **0.90** | 12 |
| t+29.4s cut | **6.54** | 7 |
| t+18.4s cut | 0.15 | 3 |
| **allstars, 30s, no cuts** | **max 0.19** | **max 3** |

Thresholds sit in the empty band between the two populations. Finds three of
four cuts with **zero false positives** on allstars, amateur and basketball —
including one at t+29.4s that neither ffmpeg scene detection nor inspection by
eye had catalogued, confirmed afterwards as a real cut.

The guard is `max(live tracks, detections)`, not live tracks alone: the t+21.6s
cut goes FROM a three-player close-up TO a fifteen-player wide shot.

**Limitation, for the report:** t+18.4s stays invisible. Keeper close-up to
behind-goal angle, both tight, so the scale barely moves. **A cut between two
similarly-scaled shots is not detectable from detections alone.**

---

## D24 · Possession is a state, not a per-frame argmax

Two bugs, one after the other.

**Wrong reference point.** Distance was measured from the ball to `p.x/p.y`, the
foot point. Right for football, where the ball is at the feet; systematically
wrong for basketball, where it is held at chest height, so the handler's feet sit
0.15-0.20 below the ball and a defender to the side is often nearer. Now
containment wins — ball centre inside a box means that player has it, deepest
inside as tie-break — and otherwise distance is to the nearest point on the box.
Sport-agnostic, no branch. Of frames with possession assigned, the ball is inside
the holder's box **70% in basketball, 22% in football**.

**Recomputed every frame.** The result was smoothed by a symmetric majority vote
over +/-0.40s, which at 5fps is two samples — too short to settle anything and
symmetric, so a neighbour winning one frame took the ring from someone who had
held it for a second. Basketball produced **22 spells in 30s, median 1.07s,
eight under half a second**.

Replaced with hysteresis: the holder keeps the ball until a challenger has been
the per-frame pick for `ON_BALL_STICK_S`. Swept, because every genuine turnover
is delayed by exactly that much:

| stick | basketball spells / median | football spells / median |
|---|---|---|
| 0.0s | 55 / 0.20s | 33 / 0.33s |
| **0.3s** | **16 / 1.55s** | **20 / 0.82s** |
| 0.6s | 11 / 2.70s | 14 / 1.23s (visibly laggy) |

**Tested and rejected:** matching the ball's velocity against each candidate's,
on the backlog as promising since 28 Aug. On ambiguous basketball frames the
assigned player's mismatch is 0.0262 against the best candidate's 0.0273 — no
signal, would change 10 frames of 883. Box overlap is not the problem either:
only 10% of basketball ball-frames sit inside two or more boxes.

---

## D25 · `--compact`: what was measured, and what I got wrong

**Corrected 3 Sep, same day it was written.** The original entry paired two
different experiments as consecutive rows of one table and read a single
mechanism across them:

- **A7**, 31 Aug: `--terse-schema`, **object** format, a different clip. Output
  **+11%** when descriptions were stripped.
- **basketball v2 vs rich**, 3 Sep: **array** format. Output **+6%** when
  descriptions were restored.

Opposite signs, different formats, different clips. And v2-vs-rich was not
controlled either: `basketball_v2` ran at 00:46, six hours before commit
`5683626` dropped the ball `kind` field, so it was *stripped + `kind`* against
*described − `kind`* — two changes worth ~6.8% pulling opposite ways. Prompt
tokens give it away: 2037 for v2, 1981 for both later runs.

**The one controlled pair.** `rich` vs `nc`: identical 1981 prompt tokens, same
clip, one hour apart, differing only in wire format.

| | reasoning | content | total completion | cost/call |
|---|---|---|---|---|
| object + descriptions (`nc`) | 1345 | 864 | 2209 | $0.004944 |
| array + descriptions (`rich`) | 1361 | **387** | 1748 | $0.004031 |

Reasoning moves **1.2%** — noise. Content **halves**. So the array format is the
entire saving, and *"a model given less guidance reasons longer"* has nothing
behind it. **The effect of the descriptions in the array format is unmeasured.**
Not "+11%", not "+4.5%" — unknown.

The array form does **lose per-field typing entirely**: `items` must admit
number, string and null, so nothing stops position 0 being a string where the
object form constrained `x` to `number` and `num` to `integer|null`. The
`players` description sentence is the only thing carrying that contract, which
is why it states the type of every position as well as its meaning.

### The 25 lost frames were the provider, not the schema

Reported as "5 to malformed JSON, where the stripped version had none". Wrong.
All five records:

```
frame  96  compl=0  reason=0  lat=30.7s  status=200
frame 258  compl=0  reason=0  lat=31.4s  status=200
frame 360  compl=0  reason=0  lat=30.3s  status=200
frame 462  compl=0  reason=0  lat=33.6s  status=200
frame 600  compl=0  reason=0  lat=33.1s  status=200
```

Zero completion tokens, zero reasoning tokens, HTTP 200. The model emitted
nothing; the `JSONDecodeError` is the parser choking on an empty-completion
envelope. That is the **zero-token-200** class already known and already
retryable, and every one sits at 30–34s alongside the 20 deadline losses. `nc`
ran p90 **17.6s** on the same endpoint an hour earlier; `rich` ran p90 **36.2s**.
One provider degradation, 25 frames, one cause.

### Where the money actually is

The split that should have been reported from the start:

| | tokens | $/video at flex |
|---|---|---|
| prompt | 1981 × 150 | $0.111 |
| **reasoning** | **1361 × 150** | **$0.383** |
| content | 387 × 150 | $0.109 |

**Reasoning is 77.9% of output and ~64% of the whole per-video bill.** Every
schema change made so far optimised content — the 18% slice. Reporting
`completion_tokens` as one number is what hid this: the +6% attributed to
descriptions was a 1282→1361 move in *reasoning* wearing a total-tokens
disguise. **Report the two separately from now on.** Correctness still gates — a
variant that loses players or misplaces boxes is rejected whatever it costs, and
cheapness is only ever a tiebreak among variants that are already correct. The
split is there to explain *why* a variant is cheaper, so the reason stops being
invented after the fact.

### `--compact` is NOT the cause of the marker artefact

Investigated at length on the user's report that ring fly-outs began when the
flag was introduced. **Field-for-field the two wire formats are equivalent:**
player counts identical on 143 of 150 frames, `w`/`h` percentiles identical,
zero rows with null or default coordinates, no transposition. The only
difference is `conf` quantisation, 13 distinct values to 8 — and `conf` reaches
only a no-op ByteTrack split (`HIGH_CONF = 0.50` is below every detection in
both files), an 11% Kalman nudge, and jersey-vote weighting.

**But the user's observation is correct and unexplained by the above.** At
allstars frame 12 the model put a player's foot at x=0.016 in the clean run and
**x=0.061 in the compact run — 58px apart**, and the ring is then held at that
wrong position until the track dies. The tracker behaved correctly: the bad
detection implied 0.09 frac/s, well inside the gate. **A 58px error anywhere
else lands on the player; at the frame edge it lands on grass.**

Whether `--compact` produces more of these, or this is run-to-run variance in a
non-deterministic model, is **not established**. The user counts six occurrences
per clip; three separate metrics of mine failed to reproduce that count, so the
visual count is the better evidence.

> **CLOSED by D27, 5 Sep.** The half of this that was "unexplained" is now
> explained, and the answer was on the tracker side after all — but not in the
> place looked at here. The sentence above, *"the tracker behaved correctly: the
> bad detection implied 0.09 frac/s, well inside the gate"*, contains the bug.
> The gate was expressed as a **rate**, so it divided by `dt`, and a dropped
> frame halved the score of an identical jump. The detection was bad *and* the
> gate should have caught it. `--compact` remains exonerated; the gate does not.

---

## D26 · Prompt and schema rewritten around judgement calls, not word count

> **RUN AND SHIPPED, 5 Sep.** This entry was written before any call was made;
> the paragraph below used to warn that its numbers were unmeasured. They have
> since been measured. What shipped is **v4** — this design plus the frame-edge
> rule — and the results are in D29 and D30.
>
> **The headline: v4 is cost-NEUTRAL against v2**, $0.1829 vs $0.1815 over 50
> interleaved frames, latency identical. The saving predicted here did not
> appear, because removing `conf` and `kits` removes *output* tokens and the
> bill is dominated by *reasoning* tokens, which the trimming did not touch.
> v4 shipped on correctness — the frame-edge rule and the simplified ball test
> — not on price. The real cost win was **v1 to v2** (26% on 3.7, 33% on 3.8),
> which had already happened before this entry was written.
>
> The `kits` undo instruction below is still live and still correct. It was
> never needed: clip 5 (volleyball) found its liberos unaided on the kit vote.

**The premise.** D25 showed reasoning is ~64% of the per-video bill, and nothing
in the prompt had ever been aimed at it. Reasoning scales with the number of
*decisions* a frame demands, not the length of the instructions. The old prompt
asked for eleven judgements per frame: box tightness, colour naming, number
legibility, confidence calibration, sport inference, goalkeeper identification,
official-vs-player, bench exclusion, count discipline, ball-vs-decoy, and kit
summarisation. Now seven.

**Duplication was the structural fault.** Every rule was stated twice — once in
`PROMPT` as prose, once in a schema `description`. "Scene first" was stated three
times, and only the schema's property order actually binds. `PROMPT` is now 440
characters against ~2600, and carries only what a schema cannot say: who is not
a player, and how to count. The per-field contract lives in the descriptions.

### Removed, with the evidence

| removed | evidence |
|---|---|
| **player `conf`** | 333 of **80,374** detections — **0.41%** — ever fell below ByteTrack's 0.50 split. It bought a judgement call and an output element per player for a decision it never made. Ball `conf` is **kept**: it multiplies the ball speed gate, where decoys sit at median 0.68 against 0.95 for real balls |
| **`kits` / `accent`** | The field worked (accent non-null in all but 95 of ~10k frames) but its only consumer is the renderer's ΔE ≥ 30 fallback, which has **never fired on any clip**, including the one nominated as the similar-kit case |
| **the worked coordinate example** | Constrained generation already fixes the shape; the arithmetic was for a human reader |
| **"the marker floats below their feet"** | **Retracted rationale.** The box render showed top and bottom edges correct. Leaving it in was telling the model its boxes run long — a nudge to shrink them |
| **the eleven named ball decoys** | boot, sock, glove, shinpad, bandage, sleeve, centre spot, penalty spot, arcs, lines, logos → one positive test: *above the surface, not painted on it, not worn*. Naming a distractor inside a negation raises its salience, and the clothing line moved basketball decoys only 4 → 2 while `football_cuts` kept plenty |

**Added:** one line for players cut off by the frame edge — *box only the part
you can actually see*. This is a correctness fix for a case the prompt never
addressed. It is **not** an explanation of the marker artefact: at
`compact_v2` t+0.184s the player was comfortably inside the frame, so whatever
drags that ring inward is not edge clamping. Remaining suspects there are
tracker-side.

### What it cost, measured offline

Player `conf` is **synthesised in `normalise_result`** rather than deleted from
the tracker, so the internal format is unchanged and `track.py` is untouched.
Every player gets 1.0, the modal value (24,134 detections reported it outright).
The ByteTrack low-confidence pass becomes explicitly empty, Kalman measurement
noise uniform, the jersey vote an unweighted count.

That is not free. Same detections, `kits` stripped and `conf` flattened:

| | raw tracks | identities | match rate p50 | ball drawn | players drawn |
|---|---|---|---|---|---|
| `basketball_v2` as-is | 27 | **17** | 0.909 | 883/900 | 5 / 10 / 10 |
| `kits` gone, `conf` flat | 28 | **18** | 0.909 | 883/900 | 5 / 10 / 10 |

**One extra fragment**, because the gate no longer loosens for a hesitant
detection. Everything else identical. A lower bound on the real change, since a
live run will also have different detections.

Losing `accent` degrades the ΔE fallback from two tiers to one — `resolve_team_colours`
falls through to `opposite(ca)`, which always returns a distinguishable colour.
`track.py` and `render.py` already tolerate both fields being absent, so
restoring the `kits` schema block is the whole of the undo. **Do that first if
clip 5 turns out to have similar kits.**

### How the next comparison must be run

Provider variance is the dominant noise term — p90 **17.6s vs 36.2s on the same
endpoint, one hour apart**. Running variant A as a block and variant B as a block
cannot separate a schema effect from that; it is how D25 came to publish a
confounded number. **Interleave the variants call-by-call inside a single run**,
and record reasoning and content tokens separately. Three arms × 50 frames = 150
calls ≈ **$0.20**, which also gives the report its one-variable ablation.

---

## D27 · The `dt` bug — three gates that measured a rate when they meant a distance

The single most productive finding of the project, because it explains a class of
artefact rather than one instance.

Every marker "fly-out" — a ring leaving its player and shooting across the frame —
traced to **one bad detection**, not to the tracker's model of motion. The tracker
had three independent defences against exactly that, and **all three were disarmed
by the same mistake**: each expressed "too far" as a *speed* or an *acceleration*,
dividing the displacement by `dt` (or `dt²`).

Detections are sampled at 5fps, but frames go missing — a refused call, a
straggler cut, a malformed row. When a frame is missing, `dt` doubles. An
identical jump therefore scores **half** the speed and **a quarter** the
acceleration. The gates relaxed precisely at the moment the tracker had least
evidence and was most exposed.

Measured on the `football_cuts` fly-out the user named: the jump scored
**10.5 bh/s² against a threshold of 30**. It was never close to firing.

**The fix is to stop dividing.** All three tests now measure displacement
directly, in body heights:

| constant | value | what it guards |
|---|---|---|
| `MAX_RESIDUAL_BH` | 1.2 | how far a sighting may sit from its own prediction |
| `OUTLIER_JUMP_BH` | 0.35 | the round-trip test on a single sighting |
| `BALL_JUMP_PH` | 1.19 | the ball, with `BALL_JUMP_FRAC` as a hard ceiling |

The residual gate rejects **0.164%** of sightings against the acceleration gate's
0.213% — it is *less* aggressive in total while actually catching the cases that
matter, which is what you expect when a test stops firing at random.

### Two things the unit had to get right

**Body-height normalisation.** Dividing image displacement by the apparent box
height gives a depth-invariant unit: a near player and a far player making the
same physical movement produce the same number. This is what makes one threshold
work across a broadcast wide shot and a goalkeeper close-up.

**The 16:9 correction is mandatory.** `x` is a fraction of frame *width*; `h` is a
fraction of frame *height*. Comparing them raw understates horizontal motion by
**1.778**, so a sideways sprint reads as 56% of its true size. Every body-height
figure in this document is aspect-corrected. This is the third time in the project
that a coordinate-space assumption has produced a wrong number (see D12), and it
will not be the last: **when a threshold behaves oddly, check the units before
tuning the value.**

### What the gate fixes, and what it does not

Fixes: any single sighting that lands far from where the track was, in any
direction. That covers every fly-out reported to date.

Does **not** fix: a wrong detection that lands *plausibly* — the correct distance
away, in the direction the player was already travelling. The user raised this
before it was built and the objection stands. A gate on displacement cannot
distinguish a real fast player from a convincing error; only appearance or a
second view could, and neither is available. The gate is a filter on the absurd,
not a truth test.

---

## D28 · Box aspect guard, and why it must be measured in pixels

Corrupted rows occasionally arrive as ribbons — a box wider than the frame and a
few pixels tall. They pass every existing check because their *position* is
legal.

Measured on **36,329 boxes** from the shipping configuration:

| | pixel aspect (w/h) |
|---|---|
| p50 | 0.463 |
| p99 | 0.874 |
| plausible tail (dives, slides, a diving goalkeeper) | 1.40 – 1.94 |
| corrupt rows | 5.61 – 8.44 |

`MAX_PIXEL_ASPECT = 3.0` sits in the empty band between those last two groups. It
rejects **3 boxes in 36,329 — 0.008%** — and every one is a ribbon. The user's
diving-goalkeeper case is real and is comfortably inside the guard: the widest
genuine box measured is 1.94, so the threshold has a 55% margin over the most
extreme legitimate posture found in five clips.

**The guard must be applied in PIXEL space, not fraction space.** Coordinates come
back as fractions of width and height independently, so a perfectly normal
standing player has a *fraction* aspect near 0.26 and a *pixel* aspect near 0.46.
A threshold set in the wrong space rejects real players. `validate_boxes()` takes
`frame_aspect` as an argument for exactly this reason.

---

## D29 · Gemini 3.8 Flash — measured and rejected

3.8 Flash launched at the same listed token price as 3.7 Flash, which made it look
free to adopt. It is not, because **price per token is not price per frame.**

Three arms, interleaved call-by-call in a single pool so all three saw identical
provider conditions, arm order rotated per frame to balance batch position
(the method D26 specified). 50 frames each, `hard10_allstars` at 1080p:

| arm | model | prompt | cost / 50 | p50 | p90 | reasoning tok | output tok |
|---|---|---|---|---|---|---|---|
| A | 3.7-flash | v2 | **$0.1819** | **10.8s** | 11.5s | 1090 | 1701 |
| B | 3.8-flash | v1 (old) | $0.3212 | 14.4s | 21.4s | 2400 | 3030 |
| C | 3.8-flash | v2 (new) | $0.2148 | 11.9s | 15.7s | 1434 | 2052 |

**3.8 costs 18% more and runs 10% slower than 3.7 on the identical prompt**, for
no quality difference visible in the renders. It reasons 32% harder about the same
frame. At the same *listed* price, that is a straight loss. **Not adopted.**

The run paid for itself anyway, because B vs C is a clean prompt ablation on a
model that had never seen either: **v1 → v2 cuts cost 33% and reasoning 40%** on
3.8, corroborating the same comparison on 3.7 (v1 $0.2475 vs v2 $0.1831 in
`v3_ab`, a 26% cut). The prompt rewrite is not a 3.7-specific artefact.

This is what the interleaved design buys. A blocked A/B here would have been read
against a 43% provider swing and told us nothing.

---

## D30 · Prompt variants v3, v5 and v6 — all rejected, each for a different reason

Four prompt sets were built after v2 and **only v4 shipped.** Recording the three
failures because two of them looked like wins on cost.

### v3 — integer coordinates. Cheapest, and unusable.

v3 moved coordinates from decimal fractions to integers 0–1000, on the reasoning
that `"0.121"` is three tokens and `"121"` is one. It worked as an economy:
**$0.1715 against v2's $0.1831**, 6% cheaper, with fewer reasoning tokens.

Then tracking collapsed.

**My first explanation was wrong, and it failed its own test.** I claimed tall
corrupt boxes were destroying association. Filtering all 64 suspect boxes left
**97 identities against 96** — no effect. The user's challenge (was this an
artefact of a hardcoded 0–1 assumption in *our* code?) was the right question; it
was not our code, but asking it is what forced the real measurement.

The real cause: **32.67% of v3 sightings have no counterpart within 0.05 in the
next frame, against 0.83% for v2.** A third of v3's detections are
frame-to-frame incoherent. Quantising to 1/1000 costs 1.9px at 1080p, far below
the ~100px localisation jitter — so precision was never the issue. Something about
emitting integers makes the model re-estimate rather than track, and the output
stops being a stable measurement of the same scene.

**The lesson is about the metric, not the prompt.** Cost, token counts and even
identity counts all said v3 was fine. The number that condemned it —
next-frame correspondence — is a *coherence* measure, and nothing in the standard
metric set was measuring coherence at all. This is §5's metric problem in a new
costume.

### v5 — ball candidate lists. The cost objection did not materialise; a better one did.

The user's hypothesis before spending: asking for *one* ball lets reasoning stop
at the first find, while asking for *all candidates* forces an exhaustive sweep of
the scene, and would cost far more. Correctly insisted this be tested before
anything was built on it.

| arm | prompt | cost / 50 | reasoning tok |
|---|---|---|---|
| A | v4 (one ball) | $0.1527 | 909 |
| B | v5 (up to 3 candidates) | $0.1588 | 949 |

**+4.0% cost, +4.4% reasoning.** The hypothesis did not hold — and testing it was
still the right call, because the reason it did not hold is the reason v5 is
useless: the model returns a mean of **0.94 candidates**. It is not sweeping the
scene and declining to rank; it is answering the same question and putting it in a
list. There is nothing to arbitrate between, so no downstream chooser can exist.

Rejected on capability, not on price. The pre-spend test was cheap and it changed
what we believed about the model.

### v6 — occlusion awareness. Worse, and more expensive.

v6 asked the model to declare the ball hidden rather than guess. On
`football_cuts`: **removed 2 decoys and introduced 3**, at +8.1% cost, +11%
latency and +13% reasoning. Net negative on the metric it was built for.

---

## D31 · Reasoning effort — nothing until "high", and "high" is unaffordable

Run at three levels, interleaved, on the frames where fly-outs had been reported:

| effort | cost / 9 frames | p50 | reasoning tok |
|---|---|---|---|
| none (default) | $0.0338 | 9.4s | 1152 |
| medium | $0.0345 | 9.2s | 1206 |
| high | $0.0745 | 16.3s | 3564 |

**"medium" is indistinguishable from the default** — +2% cost, +5% reasoning,
latency inside noise. Whatever the parameter nominally selects, it does not change
the model's behaviour at that setting on this task.

**"high" costs 120% more and 73% more latency** for 3.1× the reasoning tokens, and
did not fix the artefact it was run against — which was, by then, already known to
be the `dt` bug in D27 and not a perception failure at all.

Together with D25's finding that *low* effort destroys format compliance (36% of
frames in a wrong coordinate scale), the whole parameter is a dead lever on this
workload: below default it breaks, at medium it does nothing, above it prices
itself out. **Shipping at default.**

---

## D32 · Grid overlay — cost-neutral, and 24% slower

Drawing a labelled grid onto the frame before sending it, to turn localisation
into selection. Interleaved, 50 frames per arm:

| arm | overlay | cost | p50 |
|---|---|---|---|
| A | grid | $0.1805 | 14.2s |
| B | grid | $0.1766 | 14.6s |
| C | plain | $0.1812 | **11.5s** |

Cost is a wash; the grid arms are **24–27% slower**, and no better. This closes
ablation candidate **A1** from the second direction — D25 already rejected the
`--ruler` variant, and the labelled grid was the surviving half of that idea.

---

## D33 · Cut detection rebuilt, corroborated — and one bug I introduced

D23 moved cut detection onto shot scale. Two further changes this session.

**Corroboration by player count.** A hard cut usually changes how many players are
visible. `CUT_COUNT_RATIO = 2.5` — true cuts score **4.0 to 9.0**, and across
**595 cut-free frame boundaries the maximum is 1.50**. The band between 1.50 and
4.0 is empty, which is the only kind of threshold worth trusting. Kit-distribution
L1 distance (`CUT_KIT_L1 = 0.90`) is the third vote.

Result on `football_cuts`: **4 of 5 cuts detected, zero false positives** across
all five clips.

**The debounce regression, which the user caught.** `CUT_DEBOUNCE_S = 0.50`
suppresses repeat firings within half a second. I implemented it to keep the
**first** firing — so on the 21.6s cut it kept a weak precursor at frame 642 and
suppressed the real boundary at 648. Tracks retired 0.2s early and then
*interpolated across the actual cut*, which is the exact failure the detector
exists to prevent. Now keeps the **strongest** evidence in the window, not the
earliest.

Generalisable: **a debounce window must resolve by score, not by arrival order.**
First-wins debouncing silently prefers the noisiest edge of an event.

**The 8.8s cut remains undetectable and always will be from this data.** 18
players either side, median box height 0.075 → 0.080, an identical kit
distribution, and a scene description *more* similar than a typical non-cut
boundary. Every signal we have says "no cut". Reported as a limitation rather
than chased.

---

## D34 · Possession on the ground plane

D24 made possession a state. Two corrections since.

**The radius was wrong by 4×.** `ON_BALL_RADIUS_BH` was 1.6 body heights —
roughly **2.9 metres**, which is not possession, it is proximity. Now **0.4**.
The basketball clip moved 859 → 857 possession frames, so the tight radius costs
almost nothing while removing the class of error where a ball passing near a
stationary player briefly marks them. `MIN_POSSESSION_S = 0.40` still requires the
state to persist.

**Depth was being ignored, which the user identified from the amateur render.**
Image distance is not ground distance: a ball lofted above a player's head is
*close in pixels* and far in reality, so a header contest handed possession to
whoever happened to be under the flight path. `_airborne()` fits the ground plane
from the detections themselves — the bottom edge of a player box is their feet, so
box-bottom against box-height across a frame's players recovers the perspective
gradient, **R² p50 0.96** on `football_amateur`. A ball well above that plane is
airborne, and nobody has it.

This is a good example of the licensed division of labour: the model reports where
things *are*, and geometry works out what that *means*. No extra call.

---

## D35 · Kalman: size in the state kept, adaptive process noise rejected

Two changes bundled into one test produced a wash, which is uninterpretable. This
is the second time in the project I bundled and lost attribution (see D25);
separating them produced the verdict immediately.

**Kept — box size in the filter state.** The state is now 6-D
`[x, y, vx, vy, w, h]`, and `Track.w` / `Track.h` are properties delegating to the
filter rather than last-sighting copies. Apparent size is a depth cue, and every
body-height threshold in D27 divides by it — a size that jitters with each noisy
detection makes every gate jitter with it. Smoothing size stabilises the gates,
not just the drawing.

**Rejected — adaptive process noise.** `MANOEUVRE_GAIN` inflates Q when residuals
run high, on the theory that a player who just changed direction is less
predictable. Measured, then set to **0.0**. It fails for the same structural
reason auto-tuning failed on 31 Aug: the mechanism loosens the filter exactly when
detections are least trustworthy, so it helps a genuine swerve and helps a bad
detection equally. Kept in the code at zero as a documented dead end.

**Also rejected — an RTS smoother.** Two-sided smoothing is legitimate here in
principle (rendering is offline, so lookahead is free — see D36), but it is
**invalid on this filter**: `coast()` and `retro_correct()` mutate track state
outside the Kalman equations, so the stored covariances are not the ones that
produced the estimates. The backward pass would be weighting by numbers that no
longer describe anything. Fixing that means removing the two mechanisms that make
the tracker work.

---

## D36 · Rendering is offline, so lookahead is free

The single idea behind most of the render work: **nothing here is a live stream.**
Tracking and rendering run over a completed file, so a frame may legally be drawn
using information from later frames. Every item below is impossible in a streaming
design and nearly free in a batch one.

**Label collisions are resolved by movement, then by fading.** The old behaviour
faded overlapping labels, which loses information. `resolve_label_collisions()`
nudges labels **upward only** — `x` ties a label to its player and moving it
sideways breaks that association — and near players hold position while distant
ones give way.

**Offsets are eased over the whole clip.** The user watched the first version and
caught that the vertical displacement teleports: correct per frame, discontinuous
between them. `precompute_label_offsets()` now solves collisions for every frame
first, then eases each label's offset along a cubic across the clip. Maximum
per-frame movement falls from **48px to 9.35px**. This is only possible because
the whole timeline is known before the first pixel is drawn.

**Alpha floors, because a rule that deletes information is worse than clutter.**
`LABELS_MIN_ALPHA` was 0.0, which silently erased every distant label; now
**0.38**. Crowd density fading floors at **0.55**. Occlusion alpha is computed
from *resolved* rectangle overlap rather than raw centre distance, so a label that
was successfully moved out of the way is no longer punished for the collision it
no longer has.

**Motion easing.** `SMOOTH_FOLLOW_S = 0.10`, a critically damped follower — the
marker converges on the player without overshoot. Rendered both ways for the user
to choose; the eased version was kept.

**Fades are symmetric, and suppressed at cuts.** `FADE_OUT_S = FADE_IN_S = 0.30`.
The user's point: a track dying fades out, so a track being born should fade in,
or the two ends of a life do not match. Both are computed from distance to the
track's own first and last sample, and **both are disabled within one sample
interval of a camera cut** — at a hard cut the scene genuinely changes instantly
and a fade would misrepresent it as gradual.

---

## D37 · Marker restyle — nine designs built, all rejected, and the one thing kept

The user asked for a full visual overhaul with subagent review for readability and
"wow" factor. Nine styles were built behind `--style`: broadcast, spotlight,
tactical, stem, bar, reticle, halo, disc, arena. **All nine rejected** — the user's
verdict was that added geometry reads as clutter over moving footage, and that the
ring/ellipse on the ground is the right primitive. `classic` remains the default;
the alternatives are kept in the code because the report needs to show what was
tried, not only what shipped.

**Kept from the exercise: the typeface.** Six rounds of candidates, most rejected
as near-identical grotesques. `assets/fonts/BlackOpsOne.ttf` now heads
`FONT_STACK` and is **committed to the repo**, so a render on another machine
produces the same frames rather than silently falling back to a system font.

**Also rejected: a carrier-specific accent colour.** Tinting the player in
possession broke team identity — the user's words were that it was "actively
detrimental". Reverted to team colour with a derived outer accent. That derivation
then picked green on grass, because CIE76 ΔE weights lightness and a bright green
scores "far" from dark turf despite sharing its hue; `pick_carrier_accent()` now
carries an explicit hue guard.

**Identifier scheme.** Arabic numerals mean a number actually read off a shirt;
Roman numerals (`VII`, `XII`) are invented but stable. Roman was chosen over Greek
because it is immediately legible to a viewer with no key, while remaining
unmistakably distinct from a real jersey number.

---

## D38 · Where a call's time actually goes

Latency had been recorded as one number per call, and it did not add up to the
wall clock. Instrumented per section and re-run on `basketball` (150 calls):

| section | time |
|---|---|
| encode the frame to base64 | **1.41s** |
| time to first byte | 9.28s |
| streaming the answer back | 2.12s |
| parsing | ~0 |
| thread queue delay | 0.30s |

**Encoding was never in `latency_s`.** It is local CPU work done before the
request exists, so a stopwatch around the HTTP call cannot see it — yet it is
1.41s of every call, and it was the missing term in an 11-second discrepancy that
had been blamed on queueing. Thread queue delay, the actual suspect, is 0.30s.

Two consequences. **Concurrency is confirmed nearly free** — 150 calls complete in
about the time the slowest one takes, so the queue is not the constraint.
**Encoding is now the only part of the pipeline we control**, and it is 6% of a
call.

**Provider variance dominates everything.** The same clip, same configuration,
same endpoint: **37.4s once and 21.2s on a re-run — a 43% swing with no code
change.** Any latency figure from a single run is a sample from that distribution,
which is why the deliverable reports a range. It is also why every comparison in
this document that mattered was run interleaved.

**The straggler cut.** `CUT_SHARE = 0.97`, `CUT_GRACE_S = 1.5`: once 97% of calls
have returned, the rest are abandoned. Costs about 4 frames of 150, worst
resulting blind spell **0.40s** against the tracker's 0.60s coast — inside what
the tracker already survives.

### The straggler cut abandons the result, not the thread — measured 6 Sep

Volleyball was the slowest clip at 36.5s and was re-run instrumented to find out
why. The answer was two things, and neither is about volleyball.

**First, it was a bad draw.** Same clip, same configuration, **36.5s → 23.9s**,
a 34% swing. That is the second measured instance of provider variance at this
magnitude (basketball: 37.4s → 21.2s, 43%). Two clips, two directions of the
same effect, no code change either time. Provider variance is now a *measured
pattern*, not an anecdote, and it is the single largest term in this project's
latency.

**Second, and structural — the cut does not shorten the wall clock:**

```
last OK call finishes at      22.00s
last call of ANY kind at      23.92s   <- the wall
```

The two calls the cut abandoned set the wall. Frame 738 was cut at TTFB 14.28s
and its thread did not finish until 23.92s; the pool cannot tear down until the
abandoned socket closes. **1.92s of the run was spent waiting on calls it had
already given up on.**

This is very probably where the original 36.5s came from: three stragglers
instead of two, and no bound on how long an abandoned one takes to close.

The cut is still doing its real job — it stops the *tracker* waiting for data it
has decided to live without, which is why the worst blind spell stays at 0.40s.
It is simply not the latency saving it appears to be, and the earlier claim that
it "caps the tail" was too strong. **The fix is to stop joining abandoned
futures, not to cut sooner** — lowering `CUT_SHARE` abandons more results for the
same teardown wait. Not implemented: it changes `detect.py`, and all five
deliverables were produced with the current behaviour.

**Also: `ffmpeg` extraction is outside the reported wall.** It costs 2.09s once,
before any call, and the `wall` figure is measured across the call pool. Every
end-to-end time in this project is therefore ~2s longer than quoted.


---

## D39 · Ball decoys — five approaches, all measured, all rejected

The one defect that survived the project. The model occasionally returns a boot, a
sock, an advertising board or a painted mark instead of the ball, almost always at
a moment the real ball is genuinely occluded. The user supplied timestamps for
every instance across two clips, which is what made the failures measurable rather
than anecdotal.

| approach | result |
|---|---|
| **appearance** — confidence, size ratio, box aspect | decoys sit **inside** the real ball's distribution on all three. No separating surface exists in the features we have |
| **camera-compensated motion** | decoy residual **0.058** against a real-ball median of **0.053**. No separation |
| **candidate lists** (v5) | model returns a mean of **0.94** candidates — nothing to arbitrate between (D30) |
| **occlusion awareness** (v6) | removed 2 decoys, **introduced 3**, at 8–11% more cost (D30) |
| **positional recurrence** | every labelled decoy appears **once**, or twice separated by seconds. No clustering threshold can catch a singleton |

**A sixth approach was built, tested, and reverted for making things worse.** A
symmetric two-hop filter — indict a sighting if its neighbours both disagree —
looked principled and failed on real data: on `allstars` t+22.6–23.4s two decoys
*bracket* two real points, so the test indicted the truth and kept the errors.
**A local consistency test assumes errors are in the minority locally, and at
exactly the moments this fails, they are not.**

**Related correction: `STATIC_MIN_HITS` no longer exists.** I proposed tuning it;
the static-cluster filter it belonged to was **retired 28 Aug**. I had cited a
`HANDOFF` comment describing the removed design — precisely the failure
`CLAUDE.md` warns about, and the reason current-state and running-log documents
are kept in separate files. D15 above is history, not current behaviour.

**Reported, not hidden.** Five measured rejections are a more honest result than a
sixth heuristic tuned until the named timestamps happen to pass.

---

## D40 · The round-trip test asked geometry when it should have asked physics

Found by the user watching the volleyball render: the ball is hit straight up,
comes straight back down, and the top of the flight is **deleted**.

The test (both the ball's and the player's) is a pure excursion check — *b* is
far from *a*, far from *c*, and *a* is close to *c*, therefore *b* was never
there. A ball at the apex of a vertical flight has exactly that signature. The
neighbours are low and near each other; the apex is far from both. **Geometry
cannot separate a real out-and-back from a decoy excursion, because they are the
same shape.**

It surfaced on volleyball for a real reason rather than by chance: football and
basketball rarely sample a ball at the top of a purely vertical flight, and
volleyball does it constantly. A defect can be sport-specific in its *exposure*
while being general in its *cause*.

**First attempt: ask whether the trip was possible.** Both legs judged against
the ball's own depth-normalised speed gate — a constant already calibrated and
already applied six lines below, *after* this test had thrown the point away.

**That shipped, and it was too loose.** The user identified `allstars` frame 24
on sight: a decoy, now drawn. The gate is a ceiling on *any* ball motion and
deliberately generous, so using it as the exemption waved through any decoy that
happened to land within max-ball-speed. Frame 24's legs sit at **69% and 75%** of
the gate — comfortably "reachable", and wrong.

The user's framing is the important part, and it is a precision-over-recall
argument: *"getting all these successful frames doesn't matter, since we were
interpolating across them with relative success, if we end up drawing decoys as
well."* A false delete costs an interpolated span that is usually fine. A false
keep draws the marker on a boot. **The two errors are not symmetric and the
filter should not treat them as though they were.**

**The tightening is a different question, not a smaller number.** An out-and-back
excursion has exactly two physical causes — a ballistic apex and a bounce — and
**both are vertical reversals**. Gravity acts only downward; a bounce reverses
only the vertical component. Nothing decelerates a ball horizontally and returns
it within 0.4s, so a horizontal out-and-back has no mechanism and is an
association error by construction, *however slowly it happens*.

So the excursion must be **reachable AND vertical**. `dx` is a fraction of width
and `dy` of height, so `dx` is aspect-corrected before they are compared as a
direction — the same 16:9 correction as D27.

Measured on all 11 round-trip rejections:

| | excursion \|dy\|/\|dx\| | reachable | verdict |
|---|---|---|---|
| volleyball 558 — the reported bug | **15.5** | yes | **kept** |
| basketball 684 | **5.63** | yes | **kept** |
| basketball 522 | **3.78** | yes | **kept** |
| **allstars 24** | **0.285** | yes | **rejected — horizontal** |
| allstars 654 / 678 | 2.22 / 1.53 | no | rejected |
| basketball 450 / 864 | 0.53 / 0.64 | no | rejected |
| football_cuts 684 / 738 | 0.006 / 0.22 | no | rejected |
| volleyball 540 | 0.028 | no | rejected |

`EXCURSION_VERTICAL_RATIO = 2.0` — at least twice as vertical as horizontal, about
27° of vertical, generous room for projection and box-centre noise. Frame 24 sits
an **order of magnitude** below the nearest keep, so the threshold is not fitted
to it: anything from ~1.0 to ~3.5 separates the same way.

**Known thinness, stated rather than hidden.** `allstars` 654 and 678 are
vertical-ish decoys at 2.22 and 1.53 and *would* pass the direction test. They are
caught by reachability instead (2.50 and 2.49 against a 1.56 gate). Neither
condition is sufficient alone; both are required, and 654 sitting just above the
direction threshold is why.

Effect on the deliverables, against the committed versions:

| clip | ball drawn | added | removed | positions corrected |
|---|---|---|---|---|
| volleyball | 798 → **846** | +48 | 0 | the apex at t+18.6s |
| basketball | 858 → 858 | 0 | 0 | 28 frames at t+17.4s, t+22.8s |
| allstars | **unchanged** — byte-identical to pre-fix | | | |
| football_cuts, football_amateur | unchanged — byte-identical tracks | | | |

**Two lessons, and the second is the one that generalises.**

A test that measures *shape* will confuse two situations that share a shape —
which is D30 in a different costume.

And **an exemption must not be built from a ceiling.** The speed gate answers "is
any ball motion this fast impossible", which is the right question for rejecting
and the wrong one for sparing: pass rate at the ceiling is near 100% for exactly
the population you are trying to exclude. Sparing needs a condition that is
*rare* among errors, and direction is, because it appeals to a mechanism a
misdetection has no reason to obey.

---

## D41 · The straggler cut disabled, and the deadline lowered to 25s

The cut was the project's headline latency feature. It never worked, and the
instrumentation added for D38 is what exposed it.

**It abandons the result, not the thread.** The deadline is polled inside
`for chunk in r.iter_content(...)`, so a worker can only act on it when the next
chunk arrives — which means it cannot interrupt a stalled stream, the one case
it exists for, and before response headers arrive it is not consulted at all.
`with ThreadPoolExecutor(...)` then joins every worker on exit.

Measured on the instrumented volleyball run:

```
last OK call finished at    22.00s
last call of any kind at    23.92s   <- the wall
```

Frame 738 was cut at TTFB 14.28s and its thread did not return until 23.92s.
**1.92s of the run was spent waiting on calls it had already given up on.**

**There is no cost saving either, and it quietly understated the ledger.**
`rec["cost_usd"]` is assigned after the body parses; the cut returns before that.
0 of 14 abandoned calls across the whole project have a recorded cost — while the
generation completed server-side and was billed anyway. **~$0.042 paid and never
recorded**, which is one of D21's ledger holes identified.

So: no latency saving, no cost saving, frames discarded, ledger understated.
`CUT_SHARE = 1.0`, which the flag already supported as a documented off switch.

**And the deadline moves to `TIMEOUT_S = 25.0`.** The comment that stood there
said *"Do NOT drop it to 25 — 28s costs 3.3% of frames, 25s costs 32.7%"*. That
was measured when p90 was 41.6s, before the flex tier (D18) and the v1→v2 prompt
rewrite (D29). Re-measured per frame on the five deliverable runs:

| clip | slowest call | frames cut at 25s | worst blind spell |
|---|---|---|---|
| football_cuts | 20.9s | 0 | 0.20s |
| allstars | 21.2s | 0 | 0.40s |
| basketball | 18.2s | 0 | 0.40s |
| football_amateur | 21.1s | 0 | 0.40s |
| volleyball | 20.7s | 0 | 0.40s |

**25s is above the 100th percentile on all five** — it would not have fired once
— and the worst blind spell stays 0.40s against the tracker's 0.60s coast. The
margin is about 2s: the slowest call ever recorded in this configuration is
23.0s, on the bad-draw volleyball run. A worse draw than any yet seen would start
costing frames; that is the trade accepted in exchange for a real bound.

**Why the deadline works where the cut did not:** `requests` enforces it at the
socket, preemptively, without needing the worker to reach a polling point. The
cut was *cooperative* cancellation of a thread whose defining symptom is that it
never reaches a cancellation point. It could abandon the healthy and not the sick.

**Two things worth carrying forward.** A feature that is measured only against
the metric it was designed to improve will look like it works — the D19 sweep
recorded "wall saved" per cut share without ever checking whether abandoned
threads were still being joined, and they were. And a cancellation mechanism has
to be checked against the *blocked* case, not the running one.

---

## D42 · One run at 3fps — cheaper, faster, and measurably worse

Not planned as an ablation. The user asked, out of curiosity, what frame rate
would put a video under $0.30 and whether the render was worth seeing. It was,
so it is recorded as a finding rather than as a direction we were testing.

`allstars` at 3fps, the crowded case, everything else held:

| | 5fps (shipped) | 3fps |
|---|---|---|
| cost | $0.5253 | **$0.3217** |
| wall clock | 26.6s | **14.4s** |
| frames returned | 147/150 | 90/90 |
| identities drawn | 29 | **32** |
| jersey numbers read | 16 | 14 |
| markers | 15,450 | 15,010 |

**It is the only configuration ever run here that meets the original 15s
target**, and 39% cheaper. It also tracks visibly worse — the user's verdict was
that it "works in the most literal sense" and hides less of the compounding
error, which is the right way to put it.

**The cost is identity fragmentation, and the labels name it.** 3fps loses real
jersey numbers `30` and `93` and gains invented `I`, `II`, `III`, `XVIII`. Three
extra identities for the same twenty-two players means tracks are breaking. That
is D9's prediction arriving on schedule: spacing is constant, motion grows with
`dt`, so the gate spans 3.6× spacing at 5fps and 6.1× at 3fps.

The residual gate is *not* the signal it looked like — 309 refusals of 366 scored
pairs at 5fps against 183 of 202 at 3fps, 84% versus 91%. Proportionally similar.
The damage shows up in identity counts, not in gate pressure.

**Not shipped.** Detections and tracks are committed so the render reproduces
without a key; the video is not a deliverable.

**What this actually exposes is that 5fps was never searched for.** It came from
a geometric argument in D9 and was confirmed, not optimised — and 4fps has never
been run. On two data points, 4fps is where the trade sits: ~$0.42 and ~20s
against 5fps's $0.53 and 26.6s. Left undone deliberately, with the budget nearly
spent, but it is the cheapest open question in the project.

**And it pairs with the ball.** A second-pass ball recheck costs one extra round
trip, estimated 5–8s from the payload/TTFB relationship across 883 instrumented
calls (r = +0.483 for payload bytes against +0.061 for reasoning tokens; fastest
TTFB ever observed 3.95s). On 5fps that takes the video to 32–35s and blows the
target. On 3fps it lands at **20–22s, inside 25s**. The frame rate you would drop
to for cost is exactly what buys the headroom to afford the fix for the one
defect that survived the project. Untested, and the strongest remaining lead.

---

## D43 · The 3fps deficit is informational, not parametric — two free knobs, both null

Before building anything, the two constants that could plausibly explain the 3fps
fragmentation were swept. **Neither recovers it**, and that is the finding.

### Coast length — null

Hypothesis: `MAX_COAST_S = 0.60` sits between a 5fps single-miss gap (0.400s) and
a 3fps one (**0.667s**), so moving to 3fps drops the tolerance for a missed
detection from one to zero and tracks die at crossings. A clean discrete cliff.

Measured, `--coast 0.70` on the same 3fps detections:

| | labels | numeric | markers |
|---|---|---|---|
| coast 0.60 | 32 | 14 | 15,010 |
| coast 0.70 | 32 | 14 | 15,000 |

**Identical label sets.** Ten markers of difference in fifteen thousand. The
cliff is real arithmetic and it is not what is costing us.

### Association gate — worse, not null

Hypothesis: `MAX_RESIDUAL_BH = 1.2` is a *displacement*, so at 3fps it represents
a 1.67× slower physical speed than at 5fps and refuses real matches. The tracker
reported 183 refusals of 208 scored pairs, which looked damning.

| `MAX_RESIDUAL_BH` | identities |
|---|---|
| 1.2 (shipped) | **32** |
| 1.6 | 36 |
| 2.0 | 35 |
| 2.4 | 35 |

**Loosening it makes fragmentation worse.** The mechanism is not subtle in
hindsight: a loose gate admits *wrong* matches, the mismatched track carries on
with the wrong player, the real player is left unmatched, and a new track is born
for them. **A bad match fragments exactly as effectively as a missed one**, so
both ends of the gate's range produce births and the 1.2 setting is already near
a local optimum. The 183 refusals were the gate working, not failing.

*(A note on units: scaling this constant by the CONFIGURED sample interval would
be legitimate; scaling by the OBSERVED gap is the dt bug D27 removed. The sweep
above did the former and it still failed, so the distinction did not matter here.)*

### What that leaves

Two free parameters, two failures. **The 3fps loss survives loosening both the
coast and the gate, which means it is not a tuning artefact — it is a deficit of
information between anchors.** Nothing that only re-weights the existing
detections can recover it.

That is the strongest argument yet for a visual bridge, and it arrived by
elimination rather than by advocacy. `--residual` was added to `track.py` for this
sweep and is kept.

---

## D44 · Optical-flow bridging is affordable — measured, not estimated

The objection to between-anchor visual tracking was CPU: the latency budget is
15–25s for the whole pipeline and tracker time counts against it exactly like
network time. Earlier estimates put CSRT in minutes and MOSSE at 20–35s, i.e.
unaffordable.

**Measured on the real clip** — 900 frames at 1280×720, sparse Lucas-Kanade,
8 points per player, seeded from the actual 3fps anchor boxes, forward *and*
backward for the drift check:

| | |
|---|---|
| decode 900 frames + greyscale | **2.86s** |
| LK forward + backward, 1,547 player-bridges over 89 gaps | **3.51s** |
| **total added to the pipeline** | **6.37s** |

| configuration | wall clock | cost |
|---|---|---|
| 5fps, no bridge (shipped) | 26.6s | $0.5253 |
| 3fps, no bridge | 14.4s | $0.3217 |
| **3fps + bridge (projected)** | **~20.8s** | **$0.3217** |

**It fits, and it is faster and cheaper than what ships today.** The earlier
estimate was wrong because it assumed a full correlation tracker per player;
sparse LK on eight points is roughly two orders of magnitude cheaper and the
boxes come from the anchors, so the tracker never has to search.

### The forward-backward check has real signal

Track each point to the next anchor, then back, and measure how far it returns
from where it started:

| FB error | |
|---|---|
| p50 | **0.29 px** |
| p90 | 6.42 px |
| p99 | 91.0 px |
| max | 320.9 px |
| share above 2px | 20.8% |
| share above 5px | 12.2% |

**Most bridges are essentially exact and a clear minority fail badly**, which is
the distribution a discard rule wants — the failures are not spread evenly, they
are a separable tail. With 8 points per player, a per-player median over the
surviving points is robust to a few bad ones.

**This is the answer to the objection that a drifting visual tracker fails
*smoothly* and would therefore defeat every discontinuity-based gate we own.** FB
error is a self-assessment: the bridge grades its own reliability before anything
downstream trusts it. That asymmetry — one estimator carries an error bar, the
other does not — is what makes adjudication possible at all.

---

## D45 · Visual bridge, Phase 1 — the disagreement rate is low

`bridge.py` builds a second, independent estimate of where each track went
between anchors — sparse Lucas-Kanade through every source frame, graded by its
own forward-backward error — and compares it to the Hungarian solver's answer.
**It changes no output.** Phase 1 counts; it does not act.

Run on `allstars` at 3fps, 1,488 player-gaps across 89 anchor gaps:

| verdict | n | share |
|---|---|---|
| **AGREE** | 1,238 | **83.2%** |
| DISAGREE_OTHER_DET — bridge lands on a *different* detection (possible swap) | 29 | 1.9% |
| DISAGREE_NO_DET — bridge lands nowhere near a detection (drift, probably) | 97 | 6.5% |
| SOLVER_LOST — solver dropped the track, bridge found a detection | 3 | 0.2% |
| BOTH_LOST | 40 | 2.7% |
| BRIDGE_FAILED — too few points survive the FB check | 81 | 5.4% |

Cost: 3.93s tracking + 3.19s decode = **7.12s added**, consistent with D44's
6.37s. 3fps + bridge projects to ~21.5s against 5fps's 26.6s, at $0.3217 against
$0.5253.

### What the numbers actually say — and it is not what I predicted

**83% agreement is the reassuring part.** Two estimators with completely
different failure modes concur on five gaps in six, which is evidence that both
are basically working.

**But the actionable set is tiny.** Only **29 suspected swaps** in 1,488 gaps,
and only **3 rescues** where the bridge could keep a track the solver dropped. I
had predicted the rescue case would be the main prize — "the least controversial
use of the bridge is where there's nothing to arbitrate" — and it is 0.2% of
gaps. That prediction was wrong.

Against a fragmentation cost of **3 identities** (32 at 3fps vs 29 at 5fps), a
mechanism that flags 29 suspects and 3 rescues *could* be enough — but only if
those flags land on the right moments, which Phase 1 does not yet establish.

**The unreliable fraction is not small either:** 6.5% drift plus 5.4% failing
their own FB check is ~12% of bridges that must be discarded. The FB check is
doing real work, which is the point of having it — but it means the bridge is
not a clean second opinion everywhere, only in the 83%.

### Open, for Phase 2

Do the 29 swap-suspects and 3 rescues coincide with the moments where 3fps
actually loses identities? That is the question that decides whether this is
worth wiring in, and it needs the disagreement log cross-referenced against the
track births that 5fps does not have. **Not yet done. Nothing is wired in.**

---

# Legacy log — everything below predates D19

> These sections are kept verbatim as the running record. Several are
> **superseded**: D15's static-decoy filter was retired 28 Aug (see D39),
> the cost paragraph is superseded by D18 and D21, and the ablation table is
> closed out by D29-D32. Read D0-D39 above for current state.

## Session record — 31 Aug 2026

### Settled

| finding | evidence |
|---|---|
| **Resolution is free on Gemini and improves jersey numbers** | input tokens **2821 at 640, 960, 1280 AND 1920px** — flat across a 9x pixel range. Numbers 0.0% → 0.7% → 3.1% → 5.1% with width. `fetch_clips.py` had been downscaling 1080p source to 720p since day one for a latency budget that measurement had already shown didn't depend on resolution |
| **Luna bills by pixel and uses them; Gemini bills a flat tile rate** | Luna's numbers went 7.9% (720p/10fps) → 21.5% (1080p/5fps). Gemini's barely moved |
| **5fps is the floor; below it the gate spans too many players** | player spacing is CONSTANT (0.050–0.055 at every rate); motion grows with dt. gate÷spacing: 1.9x at 10fps, 3.6x at 5fps, **6.1x at 3fps**. Failure appears between 3.6 and 6.1 |
| **10fps buys nothing over 5fps** | Luna 30s: numbers 12.2% → 11.6%, identities 37 → 41 (worse), cost +$0.49. Same result on Gemini. We are not sampling-limited |
| **`--compact` is a free 23% saving** | output 2530 → 1798, latency 19.6s, ball 100%, numbers unchanged, no visible defect |
| **`gemini-3.7-flash` doubled in price mid-project** | $0.375/$1.875 → $0.750/$3.750 between 29 and 31 Aug. Almost certainly a launch price expiring: it had been at exactly half its predecessor and snapped to exact parity. Luna unchanged, listed seven weeks, priced consistently within its family |
| **Luna is reproducible; the clip is not uniform** | same 10s answered twice: positions agree p50 **0.0030** against 0.050 spacing, numbers 21.8% vs 22.2%. But the clip's middle third packs 27% more players 18% closer, and numbers there collapse to **6.0%** |
| **The cut detector was worse than useless** | 52 cuts across nine runs, every one a false positive landing on a corrupted-coordinate frame. Disabling it took identities 49 → 32. Removed |
| **A centred smoother overshoots a curve, forward** | 0.25(0)+0.5(1)+0.25(3) = 1.25 against a true 1. Under an accelerating pan every marker is pushed AHEAD of its player. Guarded by second difference: worst forward shift 0.021 → **0.0015** |

### Tried and rejected

- **`--ruler` (A1)** — worse on both models. Gemini numbers 17.7% → 14.2%, Luna 21.5% → 20.1%, reasoning *up* on both. It solves "where is this", and our failure is "what does that shirt say".
- **`--terse-schema` (A7)** — output tokens went *up* 11%, cost up 4%, and it produced the only coordinate-corruption frame in its run. The field descriptions were doing real work.
- **`--system` on Gemini** — +23% latency, no gain. On Luna it looked transformative at 10s (jump max 0.0149 vs 0.0313) and the effect **halved on 146 frames** (0.0292). Single-worst-value statistics on 50 frames are not evidence.
- **`--scene-last` (A4)** — helped Luna slightly, hurt Gemini. Opposite signs on the same change, so the scratchpad's value is model-specific.
- **Low reasoning effort** — 22% cheaper and it **destroys format compliance**: 36% of frames came back in a wrong coordinate scale, 13 of them mixing two scales inside one response. Reasoning effort controls instruction adherence, not just accuracy.
- **Auto-tuning the tracker's constants** — one fixed number beat the whole adaptive system, because the safety guards refuse to fire on the runs that most need help.
- **Shortening the coast (0.4s) and requiring re-confirmation** — both barely move marker placement: orphan markers 22 → 19 while missed detections go 18 → 23. A wash. Not applied.

### The metric problem, stated once

Four times now a number has said one thing and the video another:
qwen ranked first on read-rate while its renders were unusable; 3fps improved
"inflation" because fewer samples means fewer chances to fragment; `--system`
looked transformative on a single-worst-value statistic; and marker *count*
matched while marker *placement* did not.

**Every metric here counts events, and the artifacts are about placement and
continuity.** Watch the video, and use the numbers to locate the moment worth
watching — not to decide.

## Open / not yet decided

- ~~**First ablation: the ruler / coordinate reference.**~~ **CLOSED** — run and
  rejected. `--ruler` was worse on both models and drove reasoning tokens *up*.
  It solves "where is this"; the failure is "what does that shirt say". The
  labelled-grid variant was never run and no longer looks worth the spend.
- ~~**Model.**~~ **CLOSED by D16 and D19.** Luna is no longer the baseline.
  Shipping model is `google/gemini-3.7-flash` pinned to the flex provider tier.
  Luna remains the reference for jersey numbers (21.5% at 1080p) and is the
  fallback if the number-read rate turns out to bind.
- **Ball-visibility policy.** Interpolate through short gaps, hide the marker
  beyond some threshold. Threshold not yet chosen; needs the first real run.
- **Ball resolution.** If ball recall is poor, the fix is resolution or a crop
  around the predicted position, not prompting — the ball is the object most
  likely destroyed by tiling before the model ever sees it. *First evidence is
  encouraging: 1 for 1 at conf 0.97 on the one frame that returned.*

### ⚠ Open risk: jersey numbers are mostly unreadable at wide-shot resolution

First live frame: **2 numbers read out of 19 players (~10%)**. A player in that
shot is ~70px tall, so the number on their back is roughly 8–10px — genuinely
below what the encoder can resolve, and no prompt fixes that.

Consequences if this holds across clips:

- The **fallback path becomes the common case**, not the exception. Most players
  get a stable arbitrary id rather than their real number. The spec permits this,
  but it should be stated plainly rather than discovered by the reader.
- **Cut re-anchoring (D8) largely stops working**, because it is keyed on the
  number. Identities would restart at every cut.
- One of the two numbers read was `99`, which is plausible in an all-star game
  and equally plausible as a misread. Worth checking against the footage.

Candidate fixes, in order of cost: accept it and report the rate · a second pass
sending high-zoom crops around each detected player (many small cheap calls, the
same "different treatment for small objects" move considered for the ball) ·
weight the number vote so a single reading never brands a track.

### D15 · Ball decoys: screen coordinates, tested against the alternative

The static-decoy filter clusters ball detections by **screen** position. The
objection is sound in principle — a pitch marking is fixed on the *field*, so
under a pan it slides across the frame and its detections should scatter — and
the camera on this clip moves enough for it to matter: accumulated path 0.187 x
by 0.231 y, some 15–20× the 0.012 clustering radius.

Tested it anyway, clustering both ways on identical detections:

| frame | decoys found |
|---|---|
| **screen** | n=11 across 816 frames, n=7 across 660 — **both flagged** |
| stabilised | **none** |

So stabilising made it strictly worse. Our camera estimate is a sum of ~290
noisy per-frame votes and the error random-walks; it injects more than it
removes. The decoys also sit at stable *screen* positions across most of the
clip, which only holds while the camera is near-static at the moments the model
falls back to them.

**Known limitation:** on footage that pans hard and continuously this will miss
decoys. The drift-free fix is differential rather than cumulative — compare the
ball's per-frame screen displacement against the camera's and flag anything
moving *with* the camera — but that needs a camera estimate we can trust, and we
do not have one.

Separately: v7 regressed the ball because `STATIC_MIN_HITS` was dropped 5→3,
which started rejecting real detections (835 → 801 drawn). Back to 5.

### ⚠ Cost is a binding constraint after all

> **Superseded in part by D18 (1 Sep).** Cost is no longer binding for the
> shipping configuration: `gemini-3.7-flash` pinned to the flex tier delivers a
> full 30s video for **$0.8818** measured, against a $1.00 ceiling, at 5fps and
> 1080p with no schema trimming. The paragraph below still stands as the reason
> cost has to be measured per call rather than estimated — it is simply no longer
> the constraint that binds. Latency is: that same run took **31.0s wall**
> against a 25s acceptance target.

Measured on `luna_v2`, with OpenRouter's own per-call charge rather than a figure
carried over from the last project: **$0.9547 for one 30-second video.** The
ceiling is $1.00.

Earlier in this project I said "cost is not your problem and never was", from an
estimate of ~8 cents built on Ball Detector's 2.76c/100 images. That was wrong by
roughly 12×, because those calls were smaller images with far shorter answers.
At 2188 prompt tokens and ~2035 completion tokens per frame, 300 frames is
essentially the whole budget.

Consequences: sampling rate, resolution and reasoning effort are now cost levers
as well as latency levers, and A5/A6/A7 all pay twice.

### Deferred tweak — possession requires matching velocity

A player currently takes possession by proximity alone, so a ball passing near a
stationary player briefly marks them as on the ball. Real possession implies the
ball is travelling roughly *with* the player. Gate the on-ball assignment on the
ball's speed being close to that player's, not just the distance being small.
Not yet implemented; both speeds are already available in the tracker.

### Ablation candidates — running list

The brief wants parameters varied on purpose, one at a time, with a stated reason
for choosing them. Recording as we go; expect to run most of these.

| # | Vary | Hold constant | Hypothesis |
|---|---|---|---|
| A1 | Coordinate reference in the image (none / ruler / labelled grid) | everything | Helps *ball* precision more than *player* precision — the ball's error budget is far tighter. Grid turns localisation into selection, a VLM strength, at the cost of quantised precision. **First ablation.** |
| A2 | Input resolution (512 / 768 / 1024 / native 1280) | model, fps, prompt | Latency is flat against size, so this is nearly free in time. Expect player recall flat and *number* recall to fall off a cliff below native. |
| A3 | Model (Luna / qwen3-vl-32b / gemini-2.5-flash-lite / gemini-3-flash-preview) | prompt, fps, resolution | Luna's 15–19s per call cannot make 25s. The fast group is 3× quicker; the question is whether accuracy survives. Also deliverable #2. |
| A4 | `scene` field position in the schema (first vs last) | everything | Fields emit in schema order, so `scene` last means coordinates are produced with zero tokens spent thinking. Expect worse on crowded frames. |
| A5 | Reasoning effort (low / medium / high) | model, prompt | Reasoning is 40–80% of Luna's output tokens and varies 2.5× per frame (D4). Should be the largest single latency lever available. |
| A6 | Sampling rate (3 / 6 / 10 / 15 fps) | everything | D9 predicts association failures in crowded scenes below ~9fps and little gain above 15. Measure ID switches, not looks. |
| A7 | Schema description verbosity | everything | Prompt+schema costs 2073 input tokens per call — roughly half the image's own cost. Trimming descriptions is free money if accuracy holds. |
