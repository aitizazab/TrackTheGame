# Track the Game — method, frames, analysis

**Full version.** The brief asks for three pages; this is the complete record, so
that what gets cut is chosen rather than forgotten. Every number here comes from
`docs/run_log.jsonl`, `docs/decisions.md` or a committed detections file, and can
be re-derived without an API key.

---

## 1. The task, the constraints, and what was delivered

Thirty seconds of game footage in, the same thirty seconds annotated out: every
player marked, the two teams marked differently, the ball highlighted, and the
player in possession marked differently again. FIFA-style markers under the feet.
Per the instructor's clarification each player also carries a label above their
head in their team's colour, and **that label must not mutate or flash** — where
no number is legible, an arbitrary but stable identifier is accepted.

| constraint | target | delivered |
|---|---|---|
| clip length | 30s, 900 frames @ 30fps | 5 clips, all verified at exactly 900 |
| cost | under $1.00 per finished video | **$0.4662 mean** ✅ |
| processing time | under 15s, 25s accepted | **21–27s** ⚠ |
| output | a video, not stills | ✅ |
| every player marked | — | ✅ |
| teams distinguished | — | ✅ |
| ball highlighted | — | ✅ |
| player on the ball distinguished again | — | ✅ |
| labels stable, never mutating | — | ✅ |
| deliverables | 5 clips, 5 videos, repo, report | ✅ |

Three standing rules shaped everything. **VLMs for anything that looks at the
image, Pillow for anything that draws.** **No agent frameworks** — raw API calls
only, on the principle that you should not hide the loop before you have seen the
loop. And, from the instructor mid-project: *"computer vision methods can be used
for object tracking purposes, not for detection."* That third rule is what
licenses the entire architecture below.

---

## 2. Architecture: two layers, and why the split is where it is

**The VLM sees each frame independently and reports what is in it. Classical
geometry links those reports across time.** Nothing that looks at pixels is
classical; nothing that maintains identity is a VLM.

The model never learns that frames come in a sequence. It answers each one in
isolation and nothing it returns carries identity — no track IDs, no "the same
player as last frame". Everything that makes a marker stay attached to a person
happens afterwards, on the CPU, in milliseconds.

```
fetch_clips.py   download, cut to exactly 30.0s, force CFR 30fps, verify 900 frames
    ↓
detect.py        sample at 5fps, one VLM call per frame, fully parallel, strict JSON schema
    ↓
track.py         Kalman + Hungarian association, identity voting, teams, ball filtering
    ↓
render.py        Pillow markers composited onto all 900 source frames
```

The split is not a convenience. It is the only arrangement that satisfies the
instructor's clarification *and* the latency budget at once: a tracker operating
over already-returned boxes costs milliseconds and adds zero round trips, where
any per-frame model call for identity would multiply them. It also means stages 3
and 4 reproduce from committed data with **no API key at all**.

### Why clip normalisation is load-bearing

Source footage is routinely variable-frame-rate. Extract frames by index from a
VFR file and the timestamps drift, so boxes returned for "frame 300" get drawn
onto a different instant than the model saw. The annotation slides out of sync and
it presents as a tracking bug. Forcing CFR 30fps once, up front, removes the whole
class of problem. `football_cuts` is 23.976fps at source, so this was not
hypothetical.

Each clip is written **twice**: `<name>.mp4` at 1280×720 to render onto, and
`<name>_1080.mp4` at 1920×1080 to detect from. Coordinates come back as fractions,
so the two are independent — detection gets the resolution, the deliverable stays
about 14MB instead of 40MB.

---

## 3. Stage 1 — detection

### One frame per call, fully parallel

Not multi-frame windows. The concurrency probe showed median call time flat at
~3.9s to N=64 with zero failures, so there is no meaningful concurrency ceiling.
Windows would only amortise fixed per-call overhead, which true parallelism pays
once anyway, and would still serialise generation inside each call.

Measured on the shipping configuration: **all 150 calls dispatch within 0.66s**
and the run ends when the slowest lands. Thread queue delay is 0.30s. Concurrency
is effectively free.

### The schema, and why each field exists

Structured output via `json_schema`, strict, `additionalProperties: false`.

| field | why |
|---|---|
| `scene` | one sentence on framing, lighting and each team's shirt colour, emitted **first** as a reasoning scratchpad before any number is committed |
| `x, y, w, h` | **bounding box, as fractions.** Box because the foot point comes out derived (`x + w/2`, `y + h`) and box height is a depth cue that lets every threshold scale with apparent size. Fractions because resolution is an ablation axis and pixel coordinates would need rescaling between runs, where any bug looks like a model difference |
| `kit` | the shirt colour **as an ordinary word**, never "team A" — each call is independent, so a call asked for "team A" picks its own A and the teams shuffle between frames. Colour is observer-independent |
| `num` | jersey number, **null unless genuinely readable**. A wrong number is far worse than no number |
| `conf` | **ball only.** It multiplies the ball speed gate: decoys come back at median 0.68 against 0.95 for real balls |

Two fields were removed after measurement rather than on taste:

- **Player `conf`** — 333 of **80,374** detections (**0.41%**) ever fell below the
  0.50 split it fed. It bought a judgement call and an output element per player
  for a decision it never made. Synthesised downstream at 1.0 so the tracker's
  internal format is unchanged.
- **`role`** (`outfield`/`goalkeeper`) — appeared 0 times in `track.py` and 0
  times in `render.py`. It was emitted, validated, stored, and read by nothing.
  **The goalkeeper instruction stays in the prompt**; what went is the per-player
  field, not the requirement to report keepers.
- **`kits`/`accent`** — a per-frame summary of each team's colours. Its only
  consumer was a ΔE fallback that has never fired on any clip.

### Coordinate conventions are pinned per model, never inferred

Models do not reliably obey the fraction convention the schema asks for. Gemini
3.x *flash-lite* returns 0–1000; 3.7-flash returns fractions. A three-bucket probe
(`--probe-convention`) reports what a model actually did and **refuses rather than
guesses** when ambiguous; an unpinned model will not run.

The tell was in the data all along: `x+w` topped out at 1003 and 993 for two
independent models — right at 1000, not near 1280 — while `y+h` reached 835 and
865, impossible in a 720-pixel frame. **A ceiling that lands on 1000 in both axes
is a normalised space, not a resolution.** The JSON was perfectly valid; only the
renders were wrong.

### `max_tokens`

Raised twice, then lowered. The model reasons before emitting content and the cap
governs both together. On three consecutive live frames reasoning came in at 516,
1034 and 1306 tokens — a 2.5× swing — and at the original 1600 cap two of three
truncated mid-string. 6500 was the safe ceiling under the old verbose schema.
After the v4 rewrite, output medians sit near 1200–1550 tokens and the cap is
**4000**, still a ceiling rather than a reservation.

**Generalisable lesson, and it recurred:** a budget measured on an easy input is
not a budget. Both constants calibrated on a synthetic probe frame were wrong by
3–5× against real footage.

### Validation at the boundary

Three checks run before a detection is allowed into the tracker.

- **Coordinate scale.** Anything outside the pinned convention is rejected as
  corruption rather than clamped.
- **Box aspect**, measured in **pixel** space, capped at 3.0. Across **36,329
  boxes**: p50 0.463, p99 0.874, plausible tail 1.40–1.94 (including diving
  goalkeepers), corrupt rows 5.61–8.44. The threshold sits in the empty band and
  rejects **3 boxes in 36,329 — 0.008%**, every one a ribbon. It must be applied
  in pixel space: a standing player has a *fraction* aspect near 0.26 and a
  *pixel* aspect near 0.46, so a threshold set in the wrong space rejects real
  players.
- **Malformed compact rows** are counted, not silently dropped.

---

## 4. Stage 2 — association

Zero network calls. This is the layer the task is actually about.

### Kalman + centre distance, not IoU

IoU-based association needs overlap and dies where displacement exceeds box width
— about 5–8fps here. Centre distance only needs the correct match to be nearer
than every wrong one, which holds while displacement stays under roughly half the
spacing between players. In a crowded penalty box that spacing is 1–2m (19–38px at
our scale), giving a floor of 9–16fps for IoU against 2–3fps for centre distance.

The state is 6-D: `[x, y, vx, vy, w, h]`. Box size lives **in the filter**, not as
a copy of the last sighting, because apparent size is a depth cue and every
threshold below divides by it — a size that jitters with each noisy detection
makes every gate jitter with it.

Hungarian assignment matches predictions to detections on distance, kit colour (a
×4 penalty, not a veto), jersey number, and apparent box height. Camera motion is
estimated from the detections themselves and subtracted before comparing.

### Body-height normalisation, and the 16:9 correction

Dividing image displacement by apparent box height gives a **depth-invariant**
unit: a near player and a far player making the same physical movement produce the
same number. One threshold then works across a broadcast wide shot and a
goalkeeper close-up.

**The aspect correction is mandatory.** `x` is a fraction of frame *width*; `h` is
a fraction of frame *height*. Comparing them raw understates horizontal motion by
**1.778**, so a sideways sprint reads as 56% of its true size. This is the third
time in the project a coordinate-space assumption produced a wrong number. **When
a threshold behaves oddly, check the units before tuning the value.**

### Labels are properties of the track, not the frame

Jersey number and team colour are decided **once per track**, by majority vote
over every observation in that track's lifetime, then painted onto every frame.

**This is what delivers the no-flicker requirement.** Not smoothing, not
thresholding — a label decided once has nothing to flicker between. It is only
possible because we are **offline**: every tracker in the literature is online and
cannot use the future. We can.

Fallback when no number is ever legible: a **Roman numeral**, stable by
construction. Roman rather than Greek because it is immediately legible to a
viewer with no key while remaining unmistakably distinct from a real jersey
number. Arabic numerals therefore always mean "read off a shirt".

### Team colour resolution

Every player's `kit` word is counted across the whole clip and the two most-seen
colours are the two teams. A goalkeeper's third colour is therefore never mistaken
for a team, and a colour word that appears once by mistake cannot become one.

**This generalised to a sport nobody wrote code for.** On volleyball the vote came
back white 705, blue 700, red 135, green 104 — the minority-colour pattern that
identifies goalkeepers in football correctly identified liberos in a different
sport, and the referee on the stand was never detected at all.

If two kits are too close to tell apart, team B switches to the hue opposite team
A's. "Too close" is **CIE76 ΔE ≥ 30 in Lab space, not RGB distance** — RGB
disagrees with human vision badly enough to matter, and navy and black are far
apart in RGB and nearly identical on a floodlit pitch. **This fallback has never
fired**, on any clip, so the branch is unexercised.

### The `dt` bug — the single most productive finding

Every marker "fly-out" — a ring leaving its player and shooting across the frame —
traced to **one bad detection**, not to the tracker's model of motion. The tracker
had three independent defences against exactly that, and **all three were disarmed
by the same mistake**: each expressed "too far" as a *speed* or an *acceleration*,
dividing displacement by `dt` or `dt²`.

Detections are sampled at 5fps, but frames go missing — a refused call, a
malformed row. When a frame is missing, `dt` doubles, so an identical jump scores
**half** the speed and **a quarter** the acceleration. The gates relaxed precisely
when the tracker had least evidence and was most exposed. Measured on one reported
fly-out: it scored **10.5 bh/s² against a threshold of 30**. Never close to firing.

The fix is to stop dividing. All three now measure displacement directly:

| constant | value | guards |
|---|---|---|
| `MAX_RESIDUAL_BH` | 1.2 | how far a sighting may sit from its own prediction |
| `OUTLIER_JUMP_BH` | 0.35 | the round-trip test on a single sighting |
| `BALL_JUMP_PH` | 1.19 | the ball, with a flat fraction as a hard ceiling |

The residual gate rejects **0.164%** of sightings against the acceleration gate's
0.213% — *less* aggressive in total while actually catching the cases that matter,
which is what you expect when a test stops firing at random.

**What it does not fix**, stated because it was raised before the gate was built:
a wrong detection that lands *plausibly* — the right distance away, in the
direction the player was already travelling. A gate on displacement cannot
distinguish a real fast player from a convincing error. It is a filter on the
absurd, not a truth test.

### Camera cuts

Detected from the detection stream itself, on three votes:

- **shot scale** — the median box height jumps
- **player count** — `CUT_COUNT_RATIO = 2.5`. True cuts score **4.0 to 9.0**;
  across **595 cut-free frame boundaries the maximum is 1.50**. The band between
  is empty, which is the only kind of threshold worth trusting
- **kit distribution** — L1 distance ≥ 0.90

On a cut, every track is retired and interpolation is blocked across the boundary,
so no marker slides between two unrelated shots.

Result: **4 of 5 cuts on `football_cuts`, zero false positives across all five
clips.**

A debounce window (0.50s) suppresses repeat firings — and it initially kept the
**first** firing, so on the 21.6s cut it kept a weak precursor and suppressed the
real boundary, retiring tracks 0.2s early and then interpolating across the actual
cut. It now keeps the **strongest** evidence in the window. **A debounce must
resolve by score, not by arrival order**; first-wins silently prefers the noisiest
edge of an event.

### Possession

Possession is a **state**, not a per-frame argmax: it must persist
(`MIN_POSSESSION_S = 0.40`) before the marker changes.

Two corrections. The radius was **wrong by 4×** — 1.6 body heights is roughly
2.9m, which is proximity, not possession. At 0.4 the basketball clip moved 859 →
857 possession frames, so the tight radius costs almost nothing while removing the
class of error where a ball passing near a stationary player briefly marks them.

And **depth was being ignored**. Image distance is not ground distance: a ball
lofted above a player's head is close in pixels and far in reality, so a header
contest handed possession to whoever was under the flight path. The ground plane
is fitted from the detections themselves — the bottom edge of a player box is
their feet, so box-bottom against box-height across a frame's players recovers the
perspective gradient, **R² p50 0.96**. A ball well above that plane is airborne and
nobody has it.

This is the licensed division of labour working exactly as intended: the model
reports where things *are*, geometry works out what that *means*, and no extra
call is made.

### Ball filtering — precision, not recall

Recall is 97%. The failures are false positives: a pitch is covered in small white
round things — penalty spot, centre spot, painted arc, a boot, a sock — and the
model reports them confidently because they genuinely match the description.

Three geometric filters, run to a **fixed point** (up to four passes), because
removing a detection changes what its neighbours look like:

- **Flat box.** A mark painted on the turf is foreshortened vertically by an
  oblique camera and not at all horizontally, so it returns much wider than tall.
  A ball does not.
- **Round trip.** If the ball leaps away and is back next sample where it started,
  the middle reading was a decoy.
- **Over-speed.** Anything demanding more than the depth-normalised gate is a
  different object, not a fast ball. Ball `conf` **weights** this gate rather than
  gating alone.

**The round-trip test had to learn to ask physics rather than geometry.** A ball
hit straight up and falling back has *exactly* the excursion signature — far from
both neighbours, neighbours close to each other. Geometry cannot separate a real
out-and-back from a decoy because they are the same shape. It surfaced on
volleyball for a real reason: football and basketball rarely sample a ball at the
top of a vertical flight, and volleyball does it constantly. **A defect can be
sport-specific in its exposure while being general in its cause.**

The first fix — spare an excursion if both legs are within the ball's speed gate —
**was too loose and drew a decoy**. The gate is a ceiling on *any* ball motion and
deliberately generous, so it waved through any decoy landing within max-ball-speed;
the offending one sat at 69% and 75% of the gate.

The corrected rule requires the excursion to be **reachable AND vertical**. An
out-and-back has exactly two physical causes — a ballistic apex and a bounce — and
both are *vertical* reversals, because gravity acts only downward and a bounce
reverses only the vertical component. Nothing decelerates a ball horizontally and
returns it within 0.4s, so a horizontal out-and-back has no mechanism and is an
association error **however slowly it happens**. Across all 11 round-trip
rejections, excursion `|dy|/|dx|` aspect-corrected:

| kept | ratio | rejected | ratio |
|---|---|---|---|
| volleyball 558 (the reported bug) | 15.5 | allstars 24 (a decoy) | **0.285** |
| basketball 684 | 5.63 | volleyball 540 | 0.028 |
| basketball 522 | 3.78 | football_cuts 684 / 738 | 0.006 / 0.22 |
| | | allstars 654 / 678 | caught by reachability |

The decoy sits an **order of magnitude** below the nearest keep, so the 2.0
threshold is not fitted to it — anything from ~1.0 to ~3.5 separates identically.

**Known thinness, stated rather than hidden:** allstars 654 and 678 are
vertical-ish decoys at 2.22 and 1.53 and would pass the direction test. They are
caught by reachability instead. Neither condition is sufficient alone.

**The generalisable lesson is about exemptions.** A rejection rule wants a
condition that is *common* among errors; an exemption rule wants one that is
**rare** among errors. "Under max ball speed" is not rare among decoys at all.
Direction is, because it appeals to a mechanism a misdetection has no reason to
obey.

---

## 5. Stage 3 — rendering

All drawing is Pillow; **the renderer reads no pixel it did not itself write.**

The organising idea: **nothing here is a live stream.** Rendering runs over a
completed file, so a frame may legally be drawn using information from later
frames. Every item below is impossible in a streaming design and nearly free in a
batch one.

- **Flat ellipse under the feet** in the team's colour, at the box's bottom centre
  — which is why the schema asks for a box rather than a point.
- **Label above the head**, same colour, black stroke so it reads against grass,
  crowd or kit. `assets/fonts/BlackOpsOne.ttf` is **committed**, so a render on
  another machine produces the same frames rather than silently falling back to a
  system font.
- **On the ball:** the player's own marker gains a brighter fill and a derived
  outer accent. An earlier version tinted the carrier a different colour entirely
  and **destroyed team identity** — the whole point of the two-colour scheme — and
  was reverted. The derived accent then picked green on grass, because CIE76 ΔE
  weights lightness and a bright green scores "far" from dark turf despite sharing
  its hue; there is now an explicit hue guard.
- **Label collisions are resolved by movement, then by fading.** Overlapping
  labels are nudged **upward only** — `x` ties a label to its player and moving it
  sideways breaks that association — and near players hold position while distant
  ones give way.
- **Offsets are eased over the whole clip.** The first version was correct per
  frame and discontinuous between them: labels teleported. Collisions are now
  solved for every frame first, then each label's offset is eased along a cubic
  across the clip. Maximum per-frame movement falls from **48px to 9.35px**. Only
  possible because the whole timeline is known before the first pixel is drawn.
- **Alpha floors, because a rule that deletes information is worse than clutter.**
  The label alpha floor was 0.0, which silently erased every distant label; it is
  now **0.38**, and crowd-density fading floors at **0.55**. Occlusion alpha is
  computed from *resolved* rectangle overlap, so a label successfully moved out of
  the way is no longer punished for a collision it no longer has.
- **Motion easing** — a critically damped follower at 0.10s, so the marker
  converges on its player without overshoot.
- **Fades are symmetric and suppressed at cuts.** 0.30s in and 0.30s out, computed
  from distance to the track's own first and last sample. Both are disabled within
  one sample interval of a camera cut, because at a hard cut the scene genuinely
  changes instantly and a fade would misrepresent a discontinuity as gradual.

**Nine alternative marker styles were built and all nine rejected** — broadcast,
spotlight, tactical, stem, bar, reticle, halo, disc, arena. Added geometry reads as
clutter over moving footage; the ring on the ground is the right primitive. They
remain behind `--style` because a report should show what was tried, not only what
shipped.

---

## 6. The ablations

The brief asks for parameters varied on purpose, one at a time, with a stated
reason for choosing them. Each axis below was chosen because it was *suspected to
be the binding constraint at the time it was run*.

### How they had to be run

**Provider variance is the dominant noise term.** The same clip, same
configuration, same endpoint, measured twice: basketball **37.4s and 21.2s** (43%),
volleyball **36.5s and 23.9s** (34%). Running variant A as a block and variant B as
a block cannot separate a real effect from that — and doing exactly that is how
this project published a confounded result once already.

So every comparison that mattered was **interleaved call-by-call inside a single
pool**, with arm order rotated per frame to balance batch position, and reasoning
and content tokens recorded separately. All arms then see identical provider
conditions by construction.

### A1 — coordinate reference drawn on the frame

*Why:* if localisation is the weak axis, giving the model a reference should help.

`--ruler`: worse on both models. Gemini numbers 17.7% → 14.2%, and reasoning went
*up*. **Grid overlay**, interleaved, 50 frames per arm: cost-neutral and **24–27%
slower** with no accuracy gain. It solves "where is this"; our failure is "what
does that shirt say". Closed from both directions.

### A2 — input resolution

*Why:* resolution was assumed to be the main latency lever.

**Input tokens are flat at 2821 across 640/960/1280/1920 — a 9× pixel range.**
Resolution is free in tokens on this model. Jersey numbers climb 0.0% → 0.7% →
3.1% → 5.1% with width. Detection therefore runs at 1080p and rendering at 720p.

The corollary is that a *different* model may bill by pixel: Luna's numbers went
7.9% at 720p/10fps to 21.5% at 1080p/5fps, where Gemini's barely moved.

### A3 — model

*Why:* the largest single design choice available.

| model | outcome |
|---|---|
| **`google/gemini-3.7-flash`** | **shipped** — reliable, affordable on flex, adequate numbers |
| `google/gemini-3.8-flash` | launched at the *same listed token price* and costs **18% more per frame**, runs 10% slower, and reasons 32% harder about the same image. No visible quality gain. **Not adopted** |
| `openai/gpt-5.6-luna` | development baseline; best jersey numbers, but 26.8s p50. Delisted mid-project |
| `qwen/qwen3-vl-30b-a3b` | **fabricates** — 99.6% "read" jersey numbers, every box exactly 0.030×0.030, an invented squad list |
| `qwen/qwen3-vl-32b` | 28.3% read rate on paper, unusable on screen — invents an 11-ring "defensive line" that is a prior over football, not a reading of the frame |
| `mistralai/mistral-large-2512` | **best read rate measured anywhere (22.2%) and still unusable.** Under-detects, 8-word colour vocabulary including both `grey` and `gray`, match rate p50 0.600 vs 0.933, 47 identities for ~22 players |
| `moonshotai/kimi-k2.5`, `z-ai/glm-5.3-flash` | no usable detections on the convention probe, twice each |
| frontier tier (Sonnet-5, GPT-5.1, Gemini 3.1 Pro, Grok 4.5) | priced out — at 150 calls the $1 cap implies ~$3/Mtok output; all are $6–25 |

The 3.8-flash run paid for itself twice over, because its two arms are also a
clean prompt ablation on a model that had never seen either version.

### A4 — prompt version

*Why:* reasoning is ~64% of the per-video bill and nothing in the prompt had ever
been aimed at it.

Reasoning scales with the number of **decisions** a frame demands, not the length
of the instructions. v1 asked for eleven judgement calls per frame: box tightness,
colour naming, number legibility, confidence calibration, sport inference,
goalkeeper identification, official-vs-player, bench exclusion, count discipline,
ball-vs-decoy, kit summarisation. v4 asks seven.

Duplication was the structural fault — every rule was stated twice, once as prose
and once in a schema `description`, and "scene first" three times, though only the
schema's property order actually binds.

| version | result |
|---|---|
| v1 → v2 | **cost −26% on 3.7-flash, −33% on 3.8-flash.** The big win, corroborated on two models |
| v3 (integer coords 0–1000) | 6% cheaper and **unusable** — see below |
| v4 (shipped) | **cost-neutral against v2.** Shipped on correctness, not price |
| v5 (ball candidate lists) | +4.0% cost; the model returns a mean of **0.94 candidates** |
| v6 (occlusion awareness) | removed 2 decoys, **introduced 3**, at +8% cost |

**v4 being cost-neutral is worth stating plainly**, because the entry predicting it
was written before it ran. Trimming `conf` and `kits` removes *output* tokens, and
the bill is dominated by *reasoning* tokens, which the trimming never touched.

**v3 is the most instructive failure in the project.** Integer coordinates
tokenise shorter, and it was measurably cheaper. Then tracking collapsed — and
every metric we owned passed it, including identity count at 97 against 96. The
number that condemned it had to be invented: **next-frame correspondence**, the
share of sightings with no counterpart within 0.05 in the following frame.
**32.67% for v3 against 0.83% for v2.** A third of its detections were
frame-to-frame incoherent. Quantising to 1/1000 costs 1.9px at 1080p, far below
the ~100px localisation jitter, so precision was never the issue — something about
emitting integers makes the model re-estimate rather than track.

*(My first explanation for v3 — that tall corrupt boxes were destroying
association — failed its own test: filtering all 64 suspect boxes moved identities
96 → 97.)*

### A5 — reasoning effort

*Why:* reasoning is the majority of the bill, so its own control knob should be
the largest lever available.

| effort | cost / 9 frames | p50 | reasoning tokens |
|---|---|---|---|
| none (default) | $0.0338 | 9.4s | 1152 |
| medium | $0.0345 | 9.2s | 1206 |
| high | $0.0745 | 16.3s | 3564 |

**Medium is indistinguishable from the default** (+2% cost, latency inside noise).
**High costs 120% more and 73% more latency** for 3.1× the reasoning, and fixed
nothing. Separately, *low* effort **destroys format compliance**: 36% of frames
came back in a wrong coordinate scale, 13 of them mixing two scales in one
response — reasoning effort controls instruction *adherence*, not just accuracy.

A dead lever in both directions: below default it breaks, at medium it does
nothing, above it prices itself out.

### A6 — sampling rate

*Why:* the most direct cost and latency control available.

Player spacing is **constant** (0.050–0.055 at every rate) while motion grows with
`dt`, so lowering the rate should degrade association. It does. **The mechanism,
however, is the opposite of what this report previously claimed.**

The association gate takes `min()` of three limbs, and the third — a cap at 1.5×
the clip's own player spacing — **binds on every clip at every sample rate**.
Verified across seven runs: the gate ÷ spacing ratio is **1.50 by construction**,
identically at 3fps and 5fps. An earlier version of this section reported the
ratio rising 1.9× → 3.6× → 6.1× with falling frame rate; those are the *uncapped*
speed limb, a code path the shipped tracker never evaluates.

So the gate does not widen with `dt`. It stays pinned at 1.5× spacing while real
motion grows 1.67×, which makes it relatively **tighter** at 3fps — real matches
fall outside it, tracks go unmatched, and new ones are born. Scaling the cap for
the longer interval (1.5 × 5/3 = 2.5) recovers two of the three lost identities.

10fps buys nothing measurable over 5fps at twice the cost.

`allstars` — the crowded case — was then run at 3fps to see what the prediction
looks like in practice:

| | 5fps (shipped) | 3fps |
|---|---|---|
| cost | $0.5253 | **$0.3217** |
| wall clock | 26.6s | **14.4s** |
| frames returned | 147/150 | 90/90 |
| identities drawn | 29 | **32** |
| jersey numbers read | 16 | 14 |
| markers drawn | 15,450 | 15,010 |

**39% cheaper, and the only configuration ever run here that meets the original
15-second target.** It also tracks visibly worse, and the labels name the cost
precisely: 3fps *loses* real jersey numbers `30` and `93` and *gains* invented
`I`, `II`, `III` and `XVIII`. Three extra identities for the same twenty-two
players means tracks are breaking into pieces — D9's prediction arriving on
schedule. The video is committed as `outputs/videos/allstars_3fps_run.mp4`,
deliberately distinct from the five deliverables.

*(The residual gate is not the signal it first appeared to be: 309 refusals of 366
scored pairs at 5fps against 183 of 202 at 3fps — 84% versus 91%, proportionally
similar. The damage lands in identity counts, not gate pressure.)*

**What this exposes is that 5fps was never searched for.** It came from a
geometric argument and was confirmed, not optimised. 4fps has never been run, and
on the two points we have it is where the trade sits.

### A7 — schema verbosity and wire format

*Why:* prompt and schema cost ~2000 input tokens per call, roughly half the
image's own cost.

`--compact` (fixed-order arrays instead of named objects): output tokens −28.9%,
cost −26.9%, latency −12.2%, accuracy unchanged. Shipped.

`--terse-schema` (stripping `description` fields): output tokens *up* 11%, cost up
4%, and it produced the only coordinate-corruption frame in its run. The field
descriptions do real work.

### A8 — provider tier

*Why:* it turned out to be varying on its own, unmeasured, for three days.

The same model is sold at three service tiers with identical context and identical
maximum output. Default routing moved between them mid-project, which had been
recorded in the project notes as a **price increase**. It was not; the cheap tier
was live at 99.8% uptime the whole time. Solving each logged call backwards from
its own billed cost showed 511 calls at 100% flex on 28 Aug and 562 at 100%
standard on 31 Aug.

Pinned A/B, same clip, same 100 frames, only the pin differing:

| | standard | **flex** |
|---|---|---|
| $ per call | 0.011109 | **0.005294** |
| latency p50 / p90 | 22.6 / 27.6s | **11.8 / 14.2s** |
| players/frame · numbers · box aspect | 15.0 · 19.1% · 0.102/0.026 | **identical within noise** |

**Half the cost and half the latency for a routing flag**, at indistinguishable
quality — the same weights answering. The largest single effect measured anywhere
in the project.

*Caveat kept deliberately: the latency halving is not cleanly attributable,
because pinning also sets `allow_fallbacks: false`, so the batch stops making a
routing decision per call. Tier and pinning changed together.*

---

## 7. Results

Five clips, three sports. Every figure is the run that produced the committed
video, over successful calls only.

| clip | cost | wall clock | latency p50 / p90 | frames returned |
|---|---|---|---|---|
| football_cuts | $0.4727 | 22.4s | 12.3s / 15.5s | 150/150 |
| allstars | $0.5148 | 26.6s | 16.6s / 18.9s | 147/150 |
| basketball | $0.4407 | 21.2s | 11.5s / 15.1s | 149/150 |
| football_amateur | $0.4654 | 22.6s | 14.2s / 17.4s | 149/150 |
| volleyball | $0.4373 | 23.9s | 14.9s / 17.6s | 147/150 |

**Mean $0.4662 per finished video**, stable to about ±$0.04 across five clips and
three sports, against a $1.00 ceiling.

The clips were chosen to differ on the axes that stress the task: `allstars` is
the crowded broadcast wide shot (22 players); `basketball` is a tight camera with
large legible numbers; `football_amateur` is an auto-follow camera panning hard in
flat light; `football_cuts` carries five hard cuts and goalkeeper close-ups; and
`volleyball` is a third sport with a net, no goalkeepers, and a ball airborne
almost continuously — the hardest possible case for the possession rule.

### Where a call's time goes

Instrumented per section on basketball, 150 calls, medians:

| section | time |
|---|---|
| encode frame to base64 | **1.41s** |
| connect + time to first byte | 9.28s |
| stream the answer back | 2.12s |
| parse + validate | ~0 |
| thread queue delay | 0.30s |

These account for a call completely — measured against the call's own duration the
residual is **0.001s**. **Encoding was never in `latency_s`**, because it is local
CPU work done before the request exists, and it was the missing term in an
11-second discrepancy previously blamed on queueing.

They do **not** sum to the wall clock, and should not be expected to. All 150 calls
dispatch inside 0.66s and run concurrently, so the run ends when the *slowest* one
lands: the median call is 12.9s and the slowest 21.2s, which is the wall clock to
within ten milliseconds.

**The consequence governs every latency decision here: wall clock is set by the
tail, not the median.** Halving a typical call would finish the video no sooner.
That is why capping concurrency is arithmetically self-defeating, and why the
prompt rewrite cut cost 26% while barely moving wall clock. **Cost and latency are
separate problems with separate levers.**

---

## 8. What does not work, and why

A project that reports only its wins is not reporting.

### Ball decoys — the defect that survived

The model sometimes returns a boot, a sock, an advertising board or a painted mark
instead of the ball, almost always at a moment the real ball is genuinely
occluded. **Seven approaches were measured and rejected:**

| approach | result |
|---|---|
| **appearance** — confidence, size ratio, box aspect | decoys sit **inside** the real ball's distribution on all three |
| **camera-compensated motion** | decoy residual 0.058 against a real-ball median of 0.053. No separation |
| **candidate lists** (v5) | the model returns a mean of **0.94** candidates — nothing to arbitrate between |
| **occlusion awareness** (v6) | removed 2 decoys, **introduced 3**, at 8–11% more cost |
| **positional recurrence** | every labelled decoy appears **once**, or twice separated by seconds. No clustering threshold catches a singleton |
| **ball `kind` field** (which sport's ball) | all 136 basketball detections said "basketball", including every one the geometry rejected |
| **two-way outlier test** | on `allstars` t+22.6–23.4s two decoys *bracket* two real points, so a symmetric consistency test **indicts the truth** |

The pattern across the model-side attempts is consistent: the model is confidently
wrong about the ball, and asking it to grade or qualify its own answer returns the
same confidence in a new wrapper.

The pattern across the geometry-side attempts is that **every one is local** —
three consecutive points, or appearance, or recurrence — and a local test cannot
decide *which* cluster is the ball when decoys locally outnumber real detections.

### One camera cut is undetectable

The cut at 8.8s in `football_cuts` has 18 players either side, a median box height
moving 0.075 → 0.080, an identical kit distribution, and a scene description *more*
similar than a typical non-cut boundary. Every signal we have says "no cut".
Reported as a limitation rather than chased.

### Jersey numbers read at 9–32%

At broadcast distance a player is ~70px tall, so the number on their back is 8–10px
— genuinely below what the encoder resolves, and no prompt fixes that. The
**fallback path is therefore the common case**, not the exception. The spec permits
an arbitrary-but-stable identifier and that is what most players get; it is stated
rather than hidden, and Roman numerals make it visible at a glance which labels
are evidence.

### Latency misses the target

21–27s against "under 15s, 25s accepted". Diagnosed thoroughly (§7) and not fixed.
Provider variance alone is larger than any lever we control.

### The straggler cut never worked

A "dynamic p97 cut" abandoned the slowest 3% of calls once 97% had returned. It was
the project's headline latency feature and it did nothing.

The deadline is polled **inside the streaming loop**, so a worker can only act on it
when the next chunk arrives. It therefore cannot interrupt a stalled stream — the
one case it exists for — and before response headers arrive it is not consulted at
all. Instrumented on volleyball: the last useful call landed at **22.00s** and the
run did not end until **23.92s**, because `ThreadPoolExecutor` joins every worker on
exit. **1.92s spent waiting on calls already given up on.**

There was no cost saving either. `cost_usd` is assigned after the body parses and
the cut returns before that, so **0 of 14 abandoned calls have a recorded cost** —
while the generation completed server-side and was billed anyway. About **$0.042
paid and never recorded**.

So: no latency saving, no cost saving, frames discarded, ledger understated.
Disabled. The real backstop is the per-call deadline, now **25s**, which is
enforced at the socket, preemptively, without needing the worker to reach a polling
point.

**The design error is worth naming: it was cooperative cancellation of a thread
whose defining symptom is that it never reaches a cancellation point.** It could
abandon the healthy and not the sick.

---

## 9. What we got wrong

Most of this project's real findings are here, and this is the section that
demonstrates method rather than result.

### The metric problem, in both directions

**Six times a number said one thing and the video said another.** qwen ranked first
on read rate while unusable; 3fps "improved" identity inflation because fewer
samples means fewer chances to fragment; `--system` looked transformative on a
50-frame worst-value statistic that halved at 146 frames; marker *count* matched
while *placement* did not; box aspect looked fine in fraction space because 16:9
inflates it by 1.78×; and Mistral won the read-rate metric outright while failing
at the association layer.

It runs the other way too. Two defects flagged from counts — 58 frames with no
ball, and marker count falling to 13 against a median of 18 — were **both correct
behaviour**: the ball was in the keeper's hands with play dead, and there were
genuinely only 13 players in shot. **A flagged failure is a question, not a
finding, until someone looks.**

**The sharpest case is v3**, and it names the missing *kind* of metric. Cost, token
counts, reasoning tokens and identity count all passed. The number that condemned
it was **next-frame correspondence** — a *coherence* measure. Every metric we owned
counts **events**, and an event counter structurally cannot see incoherence,
because an incoherent stream contains exactly as many events as a coherent one.
**When a change looks free on every metric and wrong on screen, the missing metric
is probably about continuity between frames, not quantity within one.**

### Goalkeepers were excluded by a single word

The schema said "one entry per **outfield** player". In football that term
specifically means "not the goalkeeper", so the model correctly followed an
instruction we did not intend, and every keeper was invisible in every frame.

### A model comparison that varied four things at once

An early screen concluded that "every Gemini is near-blind to jersey numbers" and
that "newer is not better". **Both were retracted.** The read-rate claim was a
**resolution artifact** — every row ran at 1280 on the hardest clip; the same model
measured 0.2% there and **19.5%** at 1080p on ordinary footage, a ~100× spread with
no prompt change. The "newer is worse" claim compared a *flash* against
*flash-lites* (a tier difference reported as a version difference), rested on a
model that was never actually run, and treated 0.8% vs 0.6% as a difference.

**Two runs are comparable only when everything except the named variable is
pinned.** This is why every later comparison was interleaved.

### Two experiments welded into one table

`--compact` does two things — fixed-order arrays *and* stripped descriptions — and
both were introduced and measured together. Separating them, I then paired two runs
that differed in format, clip **and** an unrelated schema field, and invented a
mechanism to explain the result. The underlying reporting error was quoting
`completion_tokens`, which is reasoning *plus* content in one number: the "+6% from
descriptions" was a reasoning move wearing a total-tokens disguise. **Reasoning and
content are now reported separately.**

I had written the rule about comparisons that hold nothing constant into the
handoff document **the same morning**.

### A settled result overridden by a misremembered citation

`--compact` had been measured as a free 23% saving and written down. It was then
omitted from a signed-off run and justified by citing the "tried and rejected"
list — **which has never contained it**. One flag was confused for another, and the
finished video cost 27% more than it needed to. **Cite the line, not the memory of
the line.**

Its sibling: I proposed tuning a filter constant that had been **retired**, citing
a comment describing the removed design. This is exactly why running logs and
current-state documents are kept in separate files.

### Bundling changes and losing attribution

Twice. Three changes went into prompt v3 and the damage could not be attributed;
two Kalman changes went in together and produced a wash. Separating the second pair
produced the verdict immediately — size-in-state kept, adaptive process noise
rejected.

### An exemption built from a ceiling

The round-trip reachability fix (§4) shipped too loose and drew a decoy. Detailed
there; the lesson is that a rejection rule and an exemption rule need conditions
with opposite statistical properties, and I reused one for the other.

### ffmpeg's scene detector ranks football cuts backwards

On real footage the genuine cuts scored **0.277, 0.306, 0.237** — *below* ordinary
camera pan — while title-card wipes in the same video scored **0.46–0.94**. Every
football shot is ~70% green pitch plus crowd, so the frame histogram barely moves
across a cut while a graphic replaces the entire palette. **No threshold fixes an
inverted ordering.** A second, independent argument for associating on identity
rather than pixels.

### Subagent findings split cleanly by type

A review agent reported two bugs and one design criticism. **Both "bugs" dissolved
on checking** — one was an artefact of the crop it was given, the other described
an offset that does not exist. The **design criticism held** and changed the
render. An agent looking at output it cannot re-derive will confidently explain
artefacts of how you framed the question.

---

## 10. Limitations, honestly

- **Ball decoys persist.** Seven approaches measured and rejected. The honest
  remaining lead is an actual ball *filter* — a Kalman ball track with gating that
  scores candidates against an established trajectory, which is a **global**
  criterion where every rejected approach was local.
- **Latency 21–27s** against 25s accepted. Provider variance is larger than
  anything we control.
- **One cut in five is undetectable** from the detection stream.
- **Jersey numbers 9–32%**, so invented identifiers are the common case.
- **The ΔE colour fallback has never fired**, so that branch is unexercised. The
  similar-kit clip was never collected; volleyball was taken as clip 5 instead, on
  the grounds that a third *sport* tests more of the design than a fourth football
  clip.
- **Single-axis position offsets, cause unknown.** A decimal-rounding explanation
  was proposed, measured and **disproved** (third-decimal digit distribution shows
  no quantisation spike; `w` is *below* chance). Leading hypothesis is the centred
  smoother lagging along the direction of motion, which is single-axis by
  construction. Untested.
- **4fps was never run**, and it is where the sampling trade most likely sits.
- **The straggler cut is disabled rather than fixed.** A preemptive version —
  socket read deadlines plus `shutdown(wait=False, cancel_futures=True)` — would
  cap the tail at roughly the p97 call instead of the p100.

---

## 11. Budget

**$35 allocation, ~$29.9 recorded** across 7,882 logged calls, plus about $0.042
paid on abandoned calls that were never recorded (§8).

Every call's real charge comes from the API's own `usage.cost` and is written to
`docs/run_log.jsonl`. **Query it; do not estimate** — the running total drifted
$0.58 in one session by subtracting from memory.

The ledger was still wrong three times, and always for the same structural reason:
it was only ever as complete as the set of *writers* someone remembered to check —
a second project billing the same key, a log schema that gained `cost_usd`
mid-project, and a probe script that never recorded cost at all.

**The most expensive lesson was not a model choice but an infrastructure default.**
Unpinned provider routing silently doubled the price of every call for three days,
and it was recorded in the project notes as a price rise rather than investigated.

---

## 12. What I would do next

1. **A real ball filter** — a Kalman track for the ball with proper gating, scoring
   each candidate against a predicted position rather than against its immediate
   neighbours. Tracker-side, no API cost, and the biggest unexplored piece of the
   design.
2. **4fps**, to make the sampling choice a measured optimum rather than a confirmed
   argument.
3. **A second-pass ball recheck**, which pairs with (2) better than it does with
   5fps. One extra round trip is estimated at **5–8s** from the payload/TTFB
   relationship across 883 instrumented calls (r = +0.483 for payload bytes against
   +0.061 for reasoning tokens; fastest TTFB ever observed 3.95s). On 5fps that
   takes a video to 32–35s and blows the target; on 3fps it lands at **20–22s**.
   The frame rate you would drop to for cost is exactly what buys the headroom to
   afford the fix for the defect that survived.
4. **Preemptive straggler cancellation**, to make the tail bound real.
5. **A similar-kit clip**, to finally exercise the ΔE fallback.
