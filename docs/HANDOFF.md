# Track the Game — handoff

Written 31 Aug, rewritten 3 Sep. Read this, then `decisions.md` for the
reasoning behind each choice.

---

## 0. Start here — state as of 6 Sep

**THE PROJECT IS FEATURE-COMPLETE.** Five clips collected, five annotated videos
rendered and committed, pipeline frozen. What remains is the GitHub repo, the
report, and the LinkedIn post.

**Budget: ~$2.9 remaining.** `docs/run_log.jsonl` records $29.09 across 7,732
calls against a $35 limit, which nominally leaves $5.91 — but D21's three ledger
holes (a second project on the same key, uncosted probes) are real, so treat
~$2.9 as the working figure and **query the key endpoint before any large run**.

### The shipping configuration

```
uv run detect.py clips/<name>_1080.mp4 --fps 5 --compact     --model google/gemini-3.7-flash --prompt-version v4     --provider-order google-ai-studio/flex
uv run track.py  outputs/detections/<clip>__<tag>.json
uv run render.py outputs/tracks/<clip>__<tag>__tracks.json     --clip clips/<name>.mp4 --ball-fade
```

`v4` is the shipped prompt: the D26 rewrite, minus the `role` field, plus a
frame-edge line. Font, marker style and every threshold are defaults now — no
flags needed beyond the above.

### Measured, per finished video

| clip | cost | wall | lat p50 / p90 | frames | deliverable tag |
|---|---|---|---|---|---|
| football_cuts | $0.4727 | 22.4s | 12.3 / 15.5s | 150/150 | `cuts_v4` |
| allstars | $0.5148 | 26.6s | 16.6 / 18.9s | 147/150 | `allstars_fr_eng_v4` |
| basketball | $0.4407 | 21.2s | 11.5 / 15.1s | 149/150 | `basketball_timed` |
| football_amateur | $0.4654 | 22.6s | 14.2 / 17.4s | 149/150 | `football_amateur_v4` |
| volleyball | $0.4373 | 23.9s | 14.9 / 17.6s | 147/150 | `volleyball_timed` |

> **Percentiles are over SUCCESSFUL calls only, corrected 6 Sep.** They had
> been computed over every logged row, which silently included 150 dead Luna
> 404s filed under the `cuts_v4` tag - an aborted attempt from just before
> that model was delisted, billed at $0.00 but carrying a fast failure
> latency that pulled that row's percentiles down. Cost was never affected.
> **Filter on `ok` before quoting any latency figure from `run_log.jsonl`.**

**Mean $0.4662/video against a $1.00 cap — cost is solved.** Latency 21–27s
against "under 15s, 25s accepted" is the one constraint not met everywhere.

**Provider variance is larger than any lever we control.** Basketball ran 37.4s
and 21.2s on two runs of the identical configuration — a 43% swing, no code
change. Report latency as a range; never quote a single run as the figure.

**Where a call's time actually goes** (basketball, 150 calls, p50): encode+base64
**1.41s**, connect+TTFB 9.28s, stream body 2.12s, parse ~0. Note `latency_s`
starts *after* encoding, so **every latency figure quoted before 6 Sep is ~1.4s
per call short.** Thread queue delay is 0.30s — scheduling is not a factor.

**Those sections are complete but they are MEDIANS — do not subtract them from
the wall clock.** Measured against a call's own duration the residual is
**0.001s**, so nothing is missing inside a call. But all 150 calls dispatch
within **0.66s** and run concurrently, so the run ends when the slowest lands:
p50 call **12.91s**, slowest call **21.24s**, wall clock **21.25s**. The
difference is call-to-call variance in TTFB (9.28 → 14.33s) and streaming
(2.12 → 7.50s).

> **Design consequence, and it governs every latency decision here: wall clock
> is set by the TAIL, not the median.** Halving a typical call finishes the
> video no sooner. That is why the only lever that worked was the straggler
> cut, why `--max-concurrent` was a dead end (D19), and why the prompt rewrite
> cut cost 26% while barely moving wall clock. **Cost and latency are separate
> problems with separate levers** — a change that helps one usually does
> nothing for the other, and several days were spent before that was explicit.

### What was tried and rejected this session — all with numbers

| | verdict |
|---|---|
| **v3** 0–1000 integer coordinates | REJECTED. Saved 6.4% cost, corrupted ~6% of frames at FRAME level (63 boxes h>0.25 where v1/v2 had zero; only 1 of 973 exceeded 4× its own frame median, so no guard sees it). Tracking collapsed to 96 identities, match p50 0.587 |
| **v5** ball candidate list | REJECTED. Only +4% reasoning, so the model was already sweeping the frame — but it returns a **mean 0.94 candidates/frame** even when asked for three. Nothing to arbitrate between |
| **v6** ball-visibility judgement | REJECTED. Mechanism perfect (zero state/ball disagreements over two full clips), judgement wrong: against 7 labelled decoys it removed 2 and introduced 3, at +10.5% (cuts) to +23.5% (basketball). `partly_hidden` is a hedge — only `hidden` forces null |
| **gemini-3.8-flash** | REJECTED. Same price, +33% reasoning, +19% cost, +16% latency vs 3.7 at identical prompt. Pinned FRACTION, probed 4 Sep |
| **Grid overlay** (A1b) | NULL. Cost-neutral, reasoning slightly down, latency +24%, and **no effect on localisation jitter** — 2 grid runs sat inside the 4 plain runs' range |
| **Reasoning effort** | The provider default IS effectively medium (1152 vs 1235 reasoning). `high` is 3.1× reasoning and $1.24/video, breaking the cap, and fixed nothing |
| **Kalman adaptive process noise** | REJECTED. Isolated from the size change it added a spurious identity to basketball and cuts and improved nothing. Left at gain 0 so the ablation reproduces |
| **9 marker restyles** | REJECTED on review. Default remains the original ring |

### The ball-decoy problem is CLOSED, with five measured rejections

Do not re-propose any of these:

1. **Appearance** — decoys sit *inside* the real ball's distribution on
   confidence (0.85 vs median 0.85), size ratio (0.142 vs 0.152) and aspect
2. **Camera-compensated motion** — decoy residual 0.058 against a real-frame
   median of 0.053. Most decoys are *worn* (a boot on a moving player), so
   subtracting camera motion cannot null them
3. **Candidate multiplicity** — 0.94 candidates per frame (v5)
4. **Occlusion awareness** — net −1 decoy (v6)
5. **Positional recurrence** — every labelled decoy appears **once**, or twice
   separated by 5–13 seconds. No clustering threshold can catch a single
   sighting. NB the static-cluster filter **was retired on 28 Aug** and does not
   exist; `STATIC_MIN_HITS (RETIRED 28 Aug — the filter no longer exists)` is not a tunable, it is gone

### Open, and honestly small

**The straggler cut does not shorten the wall clock (found 6 Sep, not fixed).**
It abandons the *result* but not the *thread*, so the process waits for the
abandoned socket to close before tearing down. On the instrumented volleyball
re-run the last useful call landed at **22.00s** and the run ended at **23.92s**
— 1.92s spent waiting on calls already given up on. The cut still does its real
job (the tracker's worst blind spell stays 0.40s); it is just not a latency
saving. Fix is to stop joining abandoned futures, **not** to lower `CUT_SHARE`,
which would abandon more results for the same wait. Deferred because it changes
`detect.py` and all five deliverables were produced with current behaviour.

**Provider variance is now measured twice**, and it is the largest latency term
by a wide margin: basketball 37.4s → 21.2s (43%), volleyball 36.5s → 23.9s
(34%), identical configuration both times. Never quote a single run as *the*
figure.

**`ffmpeg` extraction (2.09s) is outside the reported wall**, which covers the
call pool only. End-to-end is ~2s longer than every figure in these documents.


- **Latency** 21–27s. ~~The p97 straggler cut is in and works~~ **DISABLED 6 Sep —
  it never worked.** It abandons the result but not the thread, so it saved no
  wall clock and no cost while discarding frames; `CUT_SHARE = 1.0`. The real
  bound is `TIMEOUT_S`, now **25.0s**, which is above the slowest call on all
  five clips and so costs nothing today. (3 calls abandoned
  on allstars, 2 on basketball). It cannot touch a call still waiting for its
  first byte, since it is checked inside the streaming loop.
- **The 8.8s cut in `football_cuts` is undetectable** from detections: 18
  players either side, median box height 0.075→0.080, identical kit mix, scene
  sentence *more* similar than a typical non-cut boundary.
- **Jersey read rate 9–32%** by shot scale. Most players carry a stable Roman
  numeral instead. Permitted; state it.

## 1. Standing constraints

- **The `.env` key is never touched.** Not read, printed, grepped, moved, or
  swept up by a wildcard. A `PreToolUse` hook blocks any Bash/PowerShell command
  containing `.env` or `OPENROUTER`, but the instruction is the boundary, not
  the hook.
- **VLMs for anything that looks at the image; Pillow for anything that draws.**
  Classical CV is permitted for *tracking* (the instructor said so explicitly),
  never for detection. Geometry over model-produced coordinates is fine.
- **No agent frameworks.** Raw API calls only.
- **Guided practice.** The user directs and must be able to defend every
  decision in an oral walkthrough. "The AI chose it" is a failing answer.
  Ask-then-explain: let them work something out, then correct in detail.
- **Deliverables:** five 30s clips collected by the user, five annotated videos
  committed, a GitHub repo with the instructor invited, a three-page report with
  a deliberate ablation, and the thinking cap.
- **Targets:** under 15s to process (25s accepted), under $1 per finished video.

## 2. Where the project actually is

**ALL FIVE CLIPS COLLECTED AND SHIPPED.** Each verified at exactly 900 frames,
CFR 30fps, in both `<name>.mp4` (1280x720, render target) and `<name>_1080.mp4`
(1920x1080, detect input). URLs and start times in `fetch_clips.py:CANDIDATES`.

| clip | what it tests | notes |
|---|---|---|
| `allstars_fr_eng` | broadcast wide, 22 players, crowding | URL never recorded — only the user can supply it |
| `basketball` | different sport, tight camera, 10 players, large legible numbers | best jersey read rate, ~30% |
| `football_amateur` | VEO auto-follow, continuous hard pan, tiny players, flat light | where fly-INS were reported |
| `football_cuts` | **four hard cuts** at 17.2s, 18.3s, 21.6s, 29.4s, plus a fifth at **8.8s that is undetectable**. Goalkeeper close-ups | the only clip exercising D8 |
| `volleyball` | **third sport**: a net, no goalkeepers, liberos in contrasting kit, ball airborne almost continuously | hardest possible possession case |

Derived test clips `first10_*` and `hard10_*` are 300 frames, not deliverables.
**Use `first10` for allstars-like testing, not `hard10`** — the user's judgement
is that it better represents what allstars actually exercises.

**The volleyball kit vote found the liberos unaided.** Colours came back white
705, blue 700, red 135, green 104 — the minority-colour pattern that identifies
goalkeepers in football generalised to a different sport's special role with no
code knowing anything about volleyball. The referee on the stand was correctly
never detected, which was the main worry for that clip.

### Budget — reconciled 1 Sep, and the ledger was incomplete

> ⚠ **Superseded by §0.** As of 6 Sep the ledger records $29.09 of a $35
> limit; treat **~$2.9** as the working figure, per D21's known holes.
> Everything in this subsection describes the state on 1 Sep and is kept for the
> method, not the numbers.

`/api/v1/auth/key` reports **limit $25, limit_remaining $5.04** — so **$19.96
used**. The project ledger totalled **$18.81**; the 6% gap is the estimated
portion below.

| | |
|---|---|
| TrackTheGame, recorded `cost_usd` | $17.05 |
| TrackTheGame, 303 successful calls never costed (est.) | $0.87 |
| Budget probes — **no cost field at all** (est.) | $0.02 |
| **BallDetector — same key, separate log, nested under `usage.cost`** | **$0.87** |
| **total** | **$18.81** |

Three structural holes, all now known: a second project on the same key, a
schema that gained `cost_usd` mid-project (297 `luna_native` calls predate it),
and `probe_budget.py` which never recorded cost at all. **`usage.cost` per call
was the right instrument; the ledger was only ever as complete as the set of
writers someone remembered to check.**

**The key is the instructor's, shared across the cohort.** Account-level
`total_credits ≈ total_usage ≈ $1445` — the *pool* is exhausted, which is not
our spend. Until it is topped up, no calls succeed regardless of the $5.04
remaining allocation. Symptoms are HTTP **402** ("would exceed your available
credits given your current in-flight requests") and **429** upstream rate limits.

**$5.04 is enough to finish, but only with `--compact`:** 4 clips $2.56 +
rebuild clip 1 $0.64 + one ablation $0.64 = **$3.84**. Without it, $5.28 — over.

Deadline was Sat 29 Aug and has been extended.

### ✅ SIGNED OFF — showcase video 1 of 5 · `gemini-3.7-flash` pinned to flex

Measured on the full 30s deliverable clip, 1080p, 5fps, 150 calls, no `--compact`:

```
uv run detect.py clips/allstars_fr_eng_1080.mp4 --fps 5 \
    --model google/gemini-3.7-flash --tag flex_30s \
    --provider-order google-ai-studio/flex
```

| | measured | target |
|---|---|---|
| **cost / 30s video** | **$0.8818** | under $1.00 ✅ |
| **wall clock** | **31.0s** | under 15s, 25s accepted ❌ |
| frames returned | 150 / 150 | — |
| latency p50 / max | 13.9s / 30.3s | — |
| identities (~22 players) | 33 | — |
| match rate p50 | **1.00** | — |
| camera cuts / degenerate frames | 0 / 0 | — |
| ball drawn | 842 / 900 frames | — |
| jersey numbers | 9.0% | — |

Render: `outputs/videos/allstars_fr_eng_1080__flex_30s.mp4`.

**Cost is solved; latency is the only failing constraint.** ⚠ **This run also
omitted `--compact` by mistake** (see D19) — with it the same video is ~$0.64 and
~12% faster, so these figures are a floor on cost, not the best available.

**Do not reach for `TIMEOUT_S`.** It is 43.0 and **never fires** — max latency was
30.3s. The wall is set by the genuinely slowest call, not by the deadline, so
lowering it cannot speed anything up; at 25s it would silently discard 49 of 150
frames. The real mechanism is upload contention (§7).

Prior candidate, for the record: `gpt-5.6-luna --system` $0.44 / 26.8s p50 / 30
identities. Its often-quoted **21.5% number-read rate is from `first10`, the easy
10s clip** — on this same 30s clip Luna reads **12.2%**. `gemini-3.7-flash
--compact` at $1.35 was on the **standard** tier: a routing artefact, not a
property of the model.

**Read rates are only comparable within one clip.** On `allstars` 30s @1080p:
Luna **12.2–12.4%**, `gemini-3.7-flash` **8.2–9.9%**. On `first10` @5fps @1080p:
Luna **21.5–21.8%**, `gemini-3.7-flash` **17.7%**. Luna leads by 20–35% on both,
and every absolute number roughly doubles between the two clips.

**Signed off by the user on 1 Sep after watching the render**, on the standard
they judge by — whether rings track correctly and whether errors are hidden, not
jersey numbers. This is **deliverable video 1 of 5**. The config above is frozen
as the baseline; further iterations are compared *against* it, not instead of it.

Committable: `git check-ignore` clears the render, the detections and the tracks.
`.gitignore` reserves `outputs/videos/*.mp4` as the deliverable — do not let this
render get swept into `outputs/videos/legacy/`.

**Both defects flagged from the numbers were false alarms**, and the user
resolved both by watching:

| flagged | reality |
|---|---|
| ball missing 58 frames (t+27.83–29.37s, t+29.63–29.97s) | the ball was **in the goalkeeper's hands** and the play had ended. Correct behaviour |
| markers drop to 13 at t+3.20–3.37s vs median 18 | there were **only 13 players in shot**. Completely correct |

See the addendum to §5 — the metric problem runs in both directions.

## 3. Measured facts worth not re-deriving

- **Resolution is FREE on Gemini** — 2821 input tokens at 640, 960, 1280 *and*
  1920px. Jersey numbers climb 0.0 → 5.1% with width. Luna bills by pixel and
  *uses* them. **CORRECTED 1 Sep — the old "7.9% → 21.5%" was a cross-clip
  comparison** (720p measured on the 30s clip, 1080p on the easy `first10`).
  Same clip, same fps, `allstars` 30s: **7.9% (720p) → 12.2% (1080p)**. Still a
  real +54% gain, not 2.7×. **Always send 1080p** — but see §7, resolution is
  free in tokens and expensive in seconds.
- **Output is 82–85% of cost** on both models. Input is a floor you cannot
  lower on Gemini.
- **5fps is the practical floor**; 3fps fails visibly. Player spacing is
  constant (~0.050) while motion grows with `dt`, so gate÷spacing goes
  1.9× (10fps) → 3.6× (5fps) → 6.1× (3fps). Failure appears between 3.6 and 6.1.
- **10fps buys nothing over 5fps** on either model.
- **Detection is 98% reliable** within an established track. Misses are only
  mildly concentrated.
- **Coordinate conventions are per-model and must be PINNED, never inferred.**
  `COORD_CONVENTION` in `detect.py`. Gemini 3.x *flash-lite* returns 0–1000;
  3.7-flash and Luna return fractions. A three-bucket probe (`--probe-convention`)
  refuses rather than guesses. An unpinned model will not run.
- **CORRECTED 1 Sep — `gemini-3.7-flash` never doubled in price.** Google sells
  the same model at three service tiers (`flex` 0.38/1.88, standard 0.75/3.75,
  `priority` 1.35/6.75). Solving all 1073 logged 1080p calls backwards from
  `usage.cost`: 28 Aug was **100% flex**, 31 Aug was **100% standard**. The
  routing moved, not the price, and flex is still live at 99.8% uptime. See D18.
  **Pin the provider on any run you intend to compare to another run** —
  `--provider-order google-ai-studio/flex`, which also sets
  `allow_fallbacks: false`. `rec["provider"]` now records what answered.
- **The catalogue's `structured: true` is an OR across providers, not a
  guarantee**, and the cheapest endpoint for a big model is routinely a 4-bit
  quantisation of it. `list_vision_models.py` reads the model record, not the
  endpoint record, so it cannot see either. Re-check prices *and* endpoints
  before trusting a cost estimate.
- **Luna is reproducible.** Same 10s answered twice agrees to p50 0.0030 against
  0.050 player spacing.

## 4. Tried and rejected — do not repeat

| | why |
|---|---|
| `--ruler` (coordinate reference drawn on frame) | worse on both models, reasoning went *up*. It solves "where is this"; our failure is "what does that shirt say" |
| `--terse-schema` | output tokens *up* 11%, and produced a coordinate-corruption frame. The field descriptions do real work |
| `--system` on Gemini | +23% latency, no gain |
| `--scene-last` | helped Luna slightly, hurt Gemini |
| **Low reasoning effort** | 22% cheaper and **destroys format compliance** — 36% of frames in a wrong coordinate scale, 13 mixing two scales in one response |
| ~~The cut detector~~ **REBUILT, and now shipping** | The *first* detector was removed: 52 cuts across nine runs, every one a false positive on a corrupted-coordinate frame, identities 49 → 32. D23 rebuilt it on shot scale and D33 added player-count and kit-distribution corroboration. Now **4 of 5 cuts on `football_cuts`, zero false positives across all five clips.** Do not read this row as "cut detection does not work" |
| Camera-motion compensation for the ball filter | three variants, all made it worse. The camera estimate is untrustworthy |
| Auto-tuning tracker constants | a single fixed number beat the whole adaptive system |
| Shortening the coast / requiring re-confirmation | orphan markers 22 → 19 but missed detections 18 → 23. A wash |
| qwen3-vl-30b-a3b | **fabricates** — 99.6% "read" jersey numbers, identical 0.030×0.030 boxes, invented squad list |
| The whole Qwen-VL line | 32B's 28.3% read-rate is not evidence of quality. On screen it invents rings — an 11-marker "defensive line" that is a prior over football, not a reading of the frame. Same failure as the 30b, better hidden. Do not re-propose it on the strength of a counter |
| `mistralai/mistral-large-2512` (1 Sep, 100 frames) | cheapest screened ($0.42/video) and **best read-rate yet at 22.2%** — and still bad. Under-detects (12 players/frame vs 15), 8-word colour vocabulary incl. both `grey` and `gray`, match rate p50 **0.600** vs 0.933, **47 identities for ~22 players**, wall 42.1s. The read-rate trap again |
| `moonshotai/kimi-k2.5`, `z-ai/glm-5.3-flash` | no usable detections on the convention probe, twice each. Both have providers lacking `structured_outputs`; unproven because `--probe-convention` discards its own errors |

### Added 5–6 Sep — all interleaved, all with numbers (D29–D32, D35, D37, D39)

| | why |
|---|---|
| **`gemini-3.8-flash`** | same *listed* token price as 3.7, **18% more cost and 10% more latency per frame** — it reasons 32% harder about the same image. No visible quality gain (D29) |
| **Prompt v3 (integer coords 0–1000)** | 6% cheaper and **unusable**: 32.67% of sightings have no counterpart within 0.05 in the next frame, against 0.83% for v2. Cost metrics all said it was fine (D30) |
| **Prompt v5 (ball candidate lists)** | only +4% cost — the price objection did not hold — but the model returns a mean of **0.94 candidates**, so there is nothing to choose between (D30) |
| **Prompt v6 (occlusion awareness)** | removed 2 decoys, **introduced 3**, at +8% cost and +11% latency (D30) |
| **Reasoning effort** | `medium` is indistinguishable from default (+2%); `high` is **+120% cost, +73% latency** and fixed nothing. `low` was already known to break format compliance. Dead lever in both directions (D31) |
| **Grid overlay** | cost-neutral and **24–27% slower**, no accuracy gain. Closes ablation A1 from the second side (D32) |
| **Adaptive process noise** (`MANOEUVRE_GAIN`) | loosens the filter exactly when detections are least trustworthy — helps a real swerve and a bad detection equally. Set to 0.0, kept as a documented dead end (D35) |
| **RTS smoother** | **invalid on this filter**, not merely unhelpful: `coast()` and `retro_correct()` mutate state outside the Kalman equations, so the stored covariances do not describe the estimates. The backward pass would weight by meaningless numbers (D35) |
| **Nine marker restyles** | broadcast, spotlight, tactical, stem, bar, reticle, halo, disc, arena — all rejected on screen. Added geometry reads as clutter over moving footage. Kept behind `--style` for the report (D37) |
| **Carrier-specific accent colour** | destroyed team identity — the whole point of the two-colour scheme. Reverted to team colour plus a derived outer accent, with a hue guard after ΔE picked green on grass (D37) |
| **Two-way outlier test on the ball** | reverted: on `allstars` t+22.6–23.4s two decoys *bracket* two real points, so a symmetric consistency test indicts the truth (D39) |

## 5. ⚠ The metric problem

**Six times a number has said one thing and the video another** (the sixth, and
the most instructive, is v3 — see the 5 Sep addendum below). qwen ranked
first on read-rate while unusable; 3fps "improved" identity inflation because
fewer samples means fewer chances to fragment; `--system` looked transformative
on a 50-frame worst-value statistic that halved at 146 frames; marker *count*
matched while *placement* did not; and box aspect looked fine in fraction space
because a 16:9 frame inflates it by 1.78×.

**Every metric here counts events; the artifacts are about placement and
continuity.** Use numbers to locate the moment worth watching, never to decide.

**When handing over a render, always name the timestamp, the specific marker,
and the predicted failure.** Never ask "does this look right".

### Addendum, 1 Sep — it runs in both directions

The five cases above are all *false negatives*: a metric looked good and the
video was bad. The `flex_30s` sign-off produced the mirror image. Two defects
were flagged from the counts — 58 frames with no ball, and marker count falling
to 13 against a median of 18 — and **both were correct behaviour**: the ball was
in the keeper's hands with the play dead, and there were genuinely only 13
players in shot.

So the counts were not lying in either case. They were doing the only job they
can do: **pointing at a moment**. The verdict on that moment came from watching
it, and would have been wrong in both directions without.

The practical consequence is that a flagged failure is a **question**, never a
finding, and it must be phrased as one when handed over. Naming a timestamp and
a predicted failure is still right — it is what let both of these be resolved in
one pass instead of being quietly written into the report as known defects. But
"58 frames with no ball" is a coordinate, not a defect, until someone looks.

### Addendum, 5 Sep — the sixth case names the missing *kind* of metric

Prompt v3 (D30) is the sharpest instance yet, because it is not a metric being
read wrongly. Cost said cheaper, token counts said cheaper, reasoning tokens said
cheaper, and even **identity count said fine — 97 against 96**. Every number in
the standard set passed. The video was unusable.

The number that condemned it had to be invented: **next-frame correspondence**,
the share of sightings with no counterpart within 0.05 in the following frame.
v3 scored **32.67%** against v2's **0.83%**.

That is a *coherence* measure, and the whole existing set counts **events** —
detections, identities, reads, frames, dollars. An event counter cannot see
incoherence, because an incoherent stream contains exactly as many events as a
coherent one. This is the same blind spot as "count matched, placement did not",
stated generally.

**So when a change looks free on every metric and wrong on screen, the missing
metric is probably about continuity between frames, not quantity within one.**
And note how it was found: the user refused my first explanation, which was
plausible, and which failed its own test the moment it was checked (filtering all
64 suspect boxes moved identities 96 → 97).

## 6. The analysis that should drive what happens next

With a constant-velocity Kalman, prediction error after `dt` is `½·a·dt²`. A
footballer accelerates ~3 m/s²; on a 68m pitch that is `a ≈ 0.044 frac/s²`.
Association survives while that stays under half the distance to the next
player, so `dt_max = √(s_min/a)` — about **1–2fps with perfect detections**.

Consecutive-miss maths says the tracker would tolerate **50% of detections going
missing** at 5fps.

**So neither frame rate nor recall is the binding constraint.** Every lever
pulled so far has been on a slack axis.

### What IS tight

~~**Position accuracy.**~~ **RETRACTED 2 Sep — the boxes are correct.**

The claim was: box aspect is 2.2–2.4 where a standing footballer is ~3.5, so
boxes are 30–40% too short and the foot point carries a systematic 0.02-unit
error. **The 3.5 reference was wrong.** It describes a person standing with arms
at their sides; a footballer in play has arms out and legs mid-stride, and a
tight box around *that* pose is ~2.2. The measurement was right and the
expectation it was compared against was invented.

Verified by rendering the raw boxes (`render.py --boxes`, cyan top edge, magenta
bottom edge) and watching the clip: **top of head and bottom of feet line up with
the box edges consistently.** Width varies a lot between boxes; both vertical
edges are reliable.

**Consequence: the foot point is already correct, and marker placement is not a
problem.** Item 6 (ask for the foot point directly) loses its accuracy rationale
entirely — see §7 for why the remaining token argument does not pay for itself.

### Two real defects the box render did expose

- **The box lags under acceleration.** Visible when a player starts or stops.
  The ellipse masks it (soft, larger than the player); the box exposes it. This
  is the centred smoother — already recorded as overshooting on a curve and
  guarded by a second-difference test. Cosmetic at ring size, real at box size.
- **Single-axis offsets — CAUSE UNKNOWN. A decimal-rounding explanation was
  proposed and then disproved.**

  The proposal: ~20% of sightings emit one coordinate at 2dp while the other
  keeps 3, snapping that axis to a ±6.4px grid. **The measurement was counting
  decimal places, which cannot tell "the model rounded" from "the true value
  happened to end in zero."** ~10% of genuine 3dp values end in 0 by chance, so
  a 10% "2dp population" is exactly what a model emitting *full* precision
  produces. Expected share with one axis apparently coarse, by chance alone:
  2 × 0.10 × 0.90 = **18%**. Observed: **20.4%**. Effectively nothing.

  The decisive test is the **third-decimal digit distribution**. Real rounding
  to 2dp would spike digit 0 hard; rounding to 0.005 steps would spike 0 and 5:

  | field | digit-0 share | chance |
  |---|---|---|
  | x | 11.2% | 10.0% |
  | y | 11.7% | 10.0% |
  | w | **8.7%** | 10.0% |
  | h | 10.8% | 10.0% |

  **No spike.** `w` is *below* chance. The distributions are mildly non-uniform
  (χ² 57–116, digit 8 over-represented in all four fields — an LLM digit
  preference, not rounding), but there is no 2dp quantisation to speak of.
  **Forcing 3dp in the prompt is therefore pointless**, and worse than pointless:
  the model already emits 3dp, so demanding precision it does not have invites
  it to fabricate the third digit. Not doing it.

  **Leading hypothesis instead — it is the same defect as the acceleration lag.**
  A centred smoother lags *along the direction of motion*. A player accelerating
  sideways lags in x with y correct; one moving toward or away from camera lags
  in y with x correct. That produces exactly "offset in one direction while the
  other axis is accurate", and it unifies the two artifacts rather than needing
  two causes. **Untested.** Testable free: correlate per-sighting residual
  direction against the track's velocity direction.

**Occlusion / projection — the user's point, and it invalidates part of the
above.** `s_min` was taken as measured *image* separation, but a 2D projection
lets two players at very different depths overlap almost completely. At a
crossing `s_min → 0` and **no frame rate resolves it from position alone.**

The signals that *can* disambiguate a crossing:

| signal | status |
|---|---|
| velocity continuity | in use, implicitly, via the Kalman prediction |
| **apparent box height = depth** | **DONE — now in the cost matrix, and in the state** |
| kit colour | used as a ×4 penalty |
| jersey number | used, but only 9–32% available |

~~**The most promising untried change is adding a box-height consistency term to
the association cost.**~~ **DONE, 5 Sep.** Box height is now a term in the
Hungarian cost and lives in the Kalman state as well (D35), so it is smoothed
rather than copied from the last sighting. It does double duty: it separates two
overlapping players at different depths — the case position cannot resolve — and
it is the divisor for every body-height threshold in D27. A jittery size makes
every gate jitter with it, which is the argument for putting it in the filter
rather than just in the cost.

## 7. Open items — CLOSED 6 Sep

Everything that was open on 3 Sep has been resolved or measured into a dead end.
Kept as a record of how each closed, because several were closed by evidence
that contradicts what was believed when they were written.

- **A. The marker fly-out artefact — FIXED.** Root cause was never the tracker.
  It was one bad detection, and the gate that should have caught it divided by
  `dt` (or `dt²`), so a dropped frame disarmed it. Three separate tests had the
  same bug. All now measure displacement in body heights. See D27.
- **B. Player boxes have no aspect guard — DONE.** Guard at 3.0 pixel aspect,
  rejecting 3 boxes in 36,329 (0.008%), every one a ribbon. Must be measured in
  PIXEL space; the fraction-space version rejects real players.
- **C. Wall clock — PARTLY.** ~~p97 straggler cut implemented and working.~~
  The cut was disabled 6 Sep as measured waste; the deadline moved to 25s. 21–27s
  remains, and provider variance (43% swing on identical runs) dominates.
- **D. Prompt bloat — DONE.** v4 shipped. The user's own prompt was never
  needed; v2's rewrite plus role removal was sufficient.

The numbered backlog that used to live here is superseded by §0. JPEG quality
(`quality=90`) remains the only untried lever, and D20 retracted the bytes model
that motivated it.

## 8. Practical notes

- Render is ~7s per 30s video (was 24s — glows are now blurred on a crop).
- `uv run track.py <detections.json>` then
  `uv run render.py <tracks.json> --clip clips/allstars_fr_eng.mp4`.
  Rendering onto the 720p clip while detecting at 1080p is deliberate: track
  coordinates are fractions, so the two resolutions are independent and the
  output file stays ~14MB instead of ~40MB.
- `render.py --ball-fade` re-enables the number-density fader. Off by default
  for debugging, on for a deliverable.
- Superseded renders live in `outputs/videos/legacy/<batch>/`, gitignored.
  ⚠ **21 superseded renders (~180 MB) are still loose in `outputs/videos/`** and
  would be committed. Moving them to `legacy/` takes the repo 239 MB → ~60 MB
  and makes the five deliverables self-evident. Not done — needs a decision.
- **Pin the provider on any run you will compare to another run** (D18):
  `--provider-order google-ai-studio/flex`, which also sets
  `allow_fallbacks: false`. `rec["provider"]` records what actually answered.
- **`--probe-convention` returns before the log write** (`detect.py:900`), so a
  failed probe reports only "no usable detections" and discards the per-call
  errors. That is why `kimi-k2.5` and `glm-5.3-flash` were dropped undiagnosed.
- **ffmpeg scene detection does not work on football** (D8): real gameplay cuts
  score 0.24–0.31, below ordinary camera pan, while title-card wipes score
  0.46–0.94. Verify cuts by eye on a contact sheet, never by threshold.
- The GitHub remote does not exist yet. First commit made 1 Sep. `.gitignore`
  excludes `.env`, `Screenshot/`, `clips/raw/`, `outputs/frames/`,
  `outputs/videos/legacy/`, and **Zeta's assignment PDF** — their document, not
  ours to publish. It stays on disk as the source of truth.

---

## 9. Prompt audit — what had earned its place

Added 3 Sep, and **acted on the same day**: this audit is what drove the rewrite
in **D26**, so it describes the PRE-rewrite prompt. Kept because it is the
justification record for what was cut. Roughly **60% of that prompt had no
experiment behind it**, and the two blocks with the clearest measurements were
the ones measured to do nothing — both are now removed. The prompt had grown by
accretion: each line defended against a specific failure, none had ever been
taken out.

**MEASURED USEFUL** — all of these survived the D26 rewrite.

| block | evidence |
|---|---|
| fractions, never pixels | D5. Resolution is an ablation axis; pixel coords would need rescaling and any bug would look like a model difference |
| `kit` as an ordinary word, never "team A" | D6. Calls are independent, so "team A" shuffles between frames |
| `num` null unless genuinely readable | D17. qwen "read" 99.6% of numbers and invented a squad list |
| GOALKEEPERS, sport-conditional | D14 — the single word "outfield" excluded every keeper. Generalisation verified: basketball 0, football 86 and 60 |
| `scene` first | A4. `--scene-last` helped Luna, hurt Gemini. Also enforced by schema property order, so the prompt line restates it |

**NO RECORDING — never tested either way.** D26 cut most of this group on the grounds that an untested line defending an unobserved failure is not free: it costs a judgement call. The exceptions kept are the officials list and the count-discipline line, which are the whole of the new prompt.

The opening line, the worked example, the TIGHT-box clause, "players on the
field of play", the officials list, "report partly hidden players", "do not pad
to a round number", the ball's tight-box line, the painted-markings line, and
"set ball to null". Two notes: the TIGHT-box clause's stated rationale (*"the
marker floats below their feet"*) was **retracted** — the boxes are correct. And
the painted-markings line has a measured *filter* behind it (37 decoys), but the
line itself was never A/B'd.

**HAS NOT EARNED ITS PLACE**

| block | measurement | outcome |
|---|---|---|
| `conf` | the `HIGH_CONF = 0.50` split is effectively a no-op — **333 of 80,374 detections (0.41%)** ever fell below it, not zero as first written. Nudges Kalman noise 11%. A 0.82 floor was tested and made ball tracking worse | **REMOVED for players** (D26). Kept for the ball, where it weights the speed gate |
| `kits` / `accent` | Feeds D11's dE>=30 fallback, which has **never fired on any clip**. Every render printed "kit colours are distinguishable; used as-is" | **REMOVED** (D26). Team colour never came from it — that is a vote over the players' own `kit` words |
| the clothing line | Added alongside the `kind` field, so its evidence is confounded. Basketball decoys 4 → 2, but cuts still has many | **REPLACED** by one positive test (D26) |

## 10. Things measured and rejected — do not re-propose

Beyond §4. Each cost real time or money to establish.

- **Ball `kind` field** (which sport's ball). All 136 basketball detections said
  "basketball", including every one the geometric filters rejected. The model
  names the sport it is watching, not the object, so the field sits downstream of
  the error. Cost 6.8% for nothing.
- **Ball confidence floor.** 0.70 measured one-for-one on Luna; 0.82 on
  gemini-3.7-flash caught none of the three named decoys and cut ball coverage
  14%. The decoys are slow — 0.10 to 0.77 frac/s against a 1.44 gate — so
  kinematics cannot see them and confidence does not separate them.
- **Rejecting balls inside player boxes.** 79% of basketball ball detections are
  inside a box. Possession *is* the ball being in a box.
- **Velocity matching for possession.** See D24.
- **Vetoing the interpolator on rejected frames.** All 17 rejections were being
  bridged and the interpolated point sat a median 0.179 from the rejected
  detection — the decoy coordinate was never drawn. The veto replaced 17 good
  positions with 17 holes.
- **Frame-bounds track death.** Never fired: no marker centre is ever outside
  [0,1]. The slide happens *inside* the frame.
- **Capping concurrency for latency.** Arithmetically worse — 3 waves at the
  fast p50 is a ~35s floor against the 31s measured.
- **ffmpeg scene detection for cuts.** Ranks football cuts backwards: real cuts
  score 0.24–0.31, below ordinary pan, while title-card wipes score 0.46–0.94.

## 11. Subagents — process rules learned the hard way

Two agents run in parallel on 3 Sep **crashed the machine**. Post-mortem:
Kernel-Power 41, BugcheckCode 0, no dump, plus `stornvme` "failed to allocate
memory" — kernel non-paged pool exhaustion, not user RAM. `render.py` spawns
**two ffmpeg processes per call**, and one agent was bisecting across seven
commits.

Rules:

- **One agent at a time.** Never two on this repo.
- **Give each an isolated workspace** and treat the project directory as
  read-only: copy `track.py` and the detections it needs into
  `%TEMP%\claude\<name>\` and run there.
- **Never let an agent run `detect.py`** — it spends real money.
- **Cap fan-out.** No loops that spawn a render per commit.
- Verify anything an agent leaves behind before trusting it. The cost-matrix
  instrumentation was kept only after checking it produced byte-identical
  tracker output with the flag off.

Agents found two things unaided that had been missed for days: the square-matrix
sentinel (D22) and the `--compact` field-by-field equivalence (D25). They are
worth using, one at a time.

**Addendum, 5 Sep — an agent's findings split cleanly by type.** The marker
review agent reported two bugs and one design criticism. **Both "bugs" dissolved
on checking**: the ghost label was an artefact of the crop I handed it, and the
claimed ball-ring offset does not exist — the ball is stored and drawn at its
centre. The **design criticism held** and changed the render.

The pattern is worth keeping: an agent looking at output it cannot re-derive
will confidently explain artefacts of *how you framed the question*. Its
judgement about what looks wrong is useful; its causal story about why is a
hypothesis to test, not a finding. Verify every claim against the source before
acting on it — which is the same rule as §5, applied to a different reporter.

## 12. Renders on disk, 6 Sep

`outputs/videos/` holds **five files and nothing else** — the deliverables:

```
FINAL_allstars_fr_eng.mp4   FINAL_basketball.mp4   FINAL_football_amateur.mp4
FINAL_football_cuts.mp4     FINAL_volleyball.mp4
```

~68 superseded experimental renders were moved to `outputs/videos/legacy/`,
which is gitignored, taking the repo-visible folder from ~1.7GB to 68MB.
`ST4_halo.mp4` — one of the nine rejected marker styles (D37) — was tracked in
here by accident and went to `legacy/` too.

**Comparison sheets and previews now live in `outputs/sheets/`**, which is
gitignored. Sixteen files: the font contact sheets, the nine-style sheets, the
ring sheet, and the before/after pairs for label collision, fade, carrier hue
and the grid/ruler overlays. **Several are candidate report figures** — promote
the ones the report cites into a tracked directory deliberately rather than
un-ignoring the folder.

### History was rewritten once, before the first push

`outputs/videos/**` was purged from all 29 commits and the five deliverables
re-added in a single commit. The directory had accumulated **40 blobs totalling
570MB to deliver 82MB** — every re-render committed a fresh copy of all five
files, and MP4 is already entropy-coded, so repacking reclaims nothing.

**`.git`: 868MB → 235MB.** Commit messages, source diffs and document history
are untouched; `git fsck --connectivity-only` is clean and all five videos
verify at 900 frames. A full pre-rewrite bundle was taken first.

**This was free only because nothing had been pushed.** Do not repeat it after
the repo is public — it needs a force-push and strands every existing clone.
If the videos are re-rendered again, that is a new blob each time: budget for it
or keep renders out of git.

**Do not re-render a tag you want to keep for comparison** — `render.py`
overwrites by stem. Use `--out`.
