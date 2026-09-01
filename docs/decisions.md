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

## D4 · `max_tokens = 4000`

**Was 1600. Raised 27 Aug after it failed on real frames.**

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
survive contact with the real batch. Derive to get the shape; measure to get the
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

### What is still open on this configuration

**Wall clock, 31.0s against a 25s acceptance target** — the only unmet hard
constraint. The diagnosis is in D18: bimodal latency driven by upload contention
at the batch tail, so the lever is `--max-concurrent`, **not** `TIMEOUT_S`.

Jersey numbers read at 9.0% on this clip against 19.5% on the easy opening 10s.
Per D7 that is survivable — a label is decided once per track by majority vote,
so a track needs one legible sighting in its lifetime, not one per frame — and
33 identities against ~22 players with zero mid-clip births says identity is
holding. It remains the weakest axis and the honest limitation for the report.

---

## D20 · Payload bytes, not resolution, and the base64 tax

**Open, not yet decided — the measurement is done, the run is not.**

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

`/api/v1/auth/key` reports **limit $25, limit_remaining $5.04** → **$19.96 used**.
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
