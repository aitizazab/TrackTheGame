# Track the Game — handoff

Written 31 Aug 2026 to start a fresh session cleanly. Read this, then
`decisions.md` for the reasoning behind each choice.

---

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

**Four clips collected of five.** All verified at exactly 900 frames, CFR 30fps,
in both `<name>.mp4` (1280x720, render target) and `<name>_1080.mp4` (detect
input). URLs and start times are recorded in `fetch_clips.py:CANDIDATES`.

| clip | source | what it tests |
|---|---|---|
| `allstars_fr_eng` | England v France, broadcast wide | **SHIPPED, video 1 of 5.** URL never recorded — only the user can supply it |
| `basketball` | 5s–35s | different sport, no goalkeeper, 10 players, large legible numbers |
| `football_amateur` | 2:56–3:26 | VEO auto-follow: continuous **pan**, tiny players, flat overcast light |
| `football_cuts` | 25s–55s | **three hard cuts** at t+17.2s, t+18.4s, t+21.8s — POV change, and the match clock jumps 02:45 → 06:24. Also a crowded goalmouth. The only footage that exercises D8 |

Derived test clips: `first10_*` (easy opening 10s) and `hard10_allstars_*`
(11–21s, hardest by crowding) — both 300 frames, not deliverables.

**Still wanted (1 of 5):** a tight/close camera where players repeatedly leave
frame — stresses track birth and death, which nothing collected so far touches.
Similar-kits is **covered** by `allstars` per the user, though note D11's
ΔE ≥ 30 fallback has therefore still never fired. A penalty was considered for
the set-piece case and rejected: too few players, too static, and the crowded
box is already inside `football_cuts`.

**Pipeline is complete and works**: `fetch_clips.py` → `detect.py` →
`track.py` → `render.py`. Plus `screen.py`, `probe_*.py`, `list_vision_models.py`,
`rank_models.py`, `repair_convention.py`.

### Budget — reconciled 1 Sep, and the ledger was incomplete

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
| The cut detector | 52 cuts across nine runs, every one a false positive on a corrupted-coordinate frame. Removed; identities 49 → 32 |
| Camera-motion compensation for the ball filter | three variants, all made it worse. The camera estimate is untrustworthy |
| Auto-tuning tracker constants | a single fixed number beat the whole adaptive system |
| Shortening the coast / requiring re-confirmation | orphan markers 22 → 19 but missed detections 18 → 23. A wash |
| qwen3-vl-30b-a3b | **fabricates** — 99.6% "read" jersey numbers, identical 0.030×0.030 boxes, invented squad list |
| The whole Qwen-VL line | 32B's 28.3% read-rate is not evidence of quality. On screen it invents rings — an 11-marker "defensive line" that is a prior over football, not a reading of the frame. Same failure as the 30b, better hidden. Do not re-propose it on the strength of a counter |
| `mistralai/mistral-large-2512` (1 Sep, 100 frames) | cheapest screened ($0.42/video) and **best read-rate yet at 22.2%** — and still bad. Under-detects (12 players/frame vs 15), 8-word colour vocabulary incl. both `grey` and `gray`, match rate p50 **0.600** vs 0.933, **47 identities for ~22 players**, wall 42.1s. The read-rate trap again |
| `moonshotai/kimi-k2.5`, `z-ai/glm-5.3-flash` | no usable detections on the convention probe, twice each. Both have providers lacking `structured_outputs`; unproven because `--probe-convention` discards its own errors |

## 5. ⚠ The metric problem

**Five times a number has said one thing and the video another.** qwen ranked
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

**Position accuracy.** The marker's foot point is `y + h`. Every model returns a
*visual* box aspect of 2.2–2.4 where a standing footballer is ~3.5 — boxes are
30–40% too short, so the foot point carries a systematic error of roughly
0.02 fraction units, which is **40% of the distance to the next player**. Not
noise: same bias, every frame, every model.

**Occlusion / projection — the user's point, and it invalidates part of the
above.** `s_min` was taken as measured *image* separation, but a 2D projection
lets two players at very different depths overlap almost completely. At a
crossing `s_min → 0` and **no frame rate resolves it from position alone.**

The signals that *can* disambiguate a crossing:

| signal | status |
|---|---|
| velocity continuity | in use, implicitly, via the Kalman prediction |
| **apparent box height = depth** | **computed, smoothed, and never used in the cost matrix** |
| kit colour | used as a ×4 penalty |
| jersey number | used, but only 8–22% available |

**The most promising untried change is adding a box-height consistency term to
the association cost.** Two overlapping players at different depths have
different apparent heights; that is exactly the case position cannot separate.

## 7. Suggested next steps, in order

> **Re-ordered 1 Sep.** Cost is solved (D18: $0.8818/video on flex). The one
> failing hard constraint is now **wall clock — 31.0s against a 25s
> acceptance target**.
>
> **The straggler model from D2 does not hold on this run, and the timeout is
> the wrong lever.** Latency is bimodal, not a thin tail: 93 calls at p50 11.7s
> and 57 calls at p50 26.4s. Cutting the deadline to 20s would drop **57 of 150
> frames (38%)**, not a handful.
>
> The slow group is **not** doing more work — correlation of latency against
> output tokens is **+0.02**, and the slow calls emit *fewer* tokens (2583 vs
> 2767). It is position in the batch:
>
> | frames submitted | lat p50 | share >20s |
> |---|---|---|
> | 0–14 | 26.1s | 53% |
> | 30–74 | 10.9–13.2s | 0–7% |
> | **120–149** | **25.3–26.4s** | **100%** |
>
> It is **upload contention**, and `detect.py:719` already measured it once: at
> 300 concurrent 1080p frames, 85 of 90 failures were
> `TimeoutError('The write operation timed out')` mid-send, and *"the same code
> at 720p had ZERO transport failures, which is the control."* At 150 concurrent
> the same constraint shows up as latency instead of failure. `t0` is set at
> `detect.py:703` — after encoding, before the POST — so **upload time is inside
> `latency_s`**.
>
> **CORRECTED — capping concurrency does not fix it, and would make it worse.**
> The 150 frames are 325 KB each: **48.8 MB that must cross the wire whatever
> the scheduling**. The 14.7s fast/slow gap over 48.8 MB implies an effective
> uplink of **~25 Mbps**, so ~15s of the 31s wall is pure upload. Capping just
> serialises it into waves:
>
> | `--max-concurrent` | waves | wall floor | verdict |
> |---|---|---|---|
> | 150 (current) | 1 | ~12s + 15s wire | 31.0s measured |
> | 96 | 2 | ~23s + wire | marginal |
> | 64 | 3 | ~35s | **worse than now** |
> | 32 | 5 | ~58s | much worse |
>
> **The lever is payload size, not concurrency and not the deadline.** Refined
> 1 Sep to include the **base64 wire tax of 33%** that the first pass missed:
> 48.4 MB of JPEG is **64.4 MB actually sent**, which puts the effective uplink
> at **~35 Mbps**, not 25.
>
> Two independent ways to cut bytes — **resolution** and **JPEG quality**:
>
> | config | KB/frame | wire MB | wire s |
> |---|---|---|---|
> | 1080p q90 (current) | 349 | 69.7 | **15.9** |
> | **1080p q85** | **285** | **57.0** | **13.0** |
> | 1080p q80 | 249 | 49.8 | 11.4 |
> | 720p q90 | 193 | 38.7 | 8.8 |
> | 720p q80 | 137 | 27.5 | 6.3 |
>
> Resolution costs jersey numbers (§3, measured). **Quality has never been
> varied at all** and might not — see step 0. On Gemini neither costs money:
> §3 measured a flat 2821 input tokens at 640, 960, 1280 *and* 1920.

0. **JPEG quality — the untested lever, and the cheapest fix available.**
   `detect.py:405` hardcodes `quality=90` and it has never been varied. Measured
   offline on 5 real frames (no API calls), calibrated to the 35 Mbps effective
   uplink implied by the 14.7s tail gap:

   | quality | KB/frame | wire MB (incl. base64) | wire s | projected wall |
   |---|---|---|---|---|
   | **90 (current)** | 349 | 69.7 | 15.9 | **27.6s** ✗ |
   | **85** | 285 | 57.0 | 13.0 | **24.7s** ✅ |
   | 80 | 249 | 49.8 | 11.4 | 23.1s ✅ |
   | 75 | 223 | 44.6 | 10.2 | 21.9s ✅ |
   | *720p @ q90* | *193* | *38.7* | *8.8* | *20.5s* |

   **q85 projects inside the 25s target at full 1080p.** One character.
   Note base64 is a **33% wire tax** — 48.4 MB of JPEG is 64.4 MB on the wire.

1. **The ablation this sets up, and it beats `--width`.** Two ways to spend the
   same byte budget, which should damage jersey numbers *differently*:
   720p @ q90 (193 KB) halves the pixels, so an 8px number becomes 4px — the
   information is **gone**. 1080p @ q80 (249 KB) keeps every pixel and adds
   ringing — the number is **noisier but still there**. Hypothesis: at a fixed
   payload, lowering quality beats lowering resolution for small-text
   legibility. Genuinely uncertain, since JPEG destroys exactly the
   high-frequency detail a small number *is*. Three configs, one variable,
   ~$1.92 with `--compact`.

2. **`max_tokens` is the 402 reservation basis.** It is 4000; with `--compact`
   the median output is 1798. Dropping to 3000 cuts the per-request hold 25% and
   directly eases the in-flight ceiling that produced the 402s. D4's warning
   still applies — a cap measured on easy input is not a cap.

   **Do not lower `TIMEOUT_S` to hit the latency target: at 25s it silently
   discards 49 of 150 frames, and it never fires today anyway.**

3. **Rebuild clip 1 with `--compact`** (~$0.64) — a settled win wrongly omitted
   from the signed-off run (D19). Then run clips 2–5. Budget: $3.84 of $5.04.
4. **`call_with_retry` handles no HTTP status errors** (`detect.py:678` covers
   only `ConnectionError`/`SSLError`/`ChunkedEncoding`/`RemoteDisconnected`).
   **Retry 429 with backoff; abort the run on 402.** Six of eight losses on
   1 Sep were plain rate limiting silently dropped as frames. Free.
5. **Check which end of the box is wrong** (free, no API). Compare box tops and
   bottoms against player positions. If tops are right and bottoms are high, the
   model is boxing torsos — a different fix from boxing loosely. **This gates
   item 6**: the user's viewing suggests the markers sit correctly under feet,
   which would mean the bottom is right, the boxes are short at the *top*, and
   the foot-point change buys nothing.
6. **Ask for the foot point directly** rather than deriving `y + h`. Measured at
   only **5.4% of output (~$0.048/video)**, so the token case is weak and cost
   is no longer binding. Do it for *accuracy*, and only if item 5 says the
   bottom edge is wrong.
7. **Box-height term in the cost matrix** (free, no API). Deferred by the user,
   correctly: association is at ceiling on `allstars` (match rate p50 **1.00**,
   zero mid-clip births), so an improvement to it is unmeasurable there. It
   becomes testable on `football_cuts` and `football_amateur`.
8. ~~**Screen a higher price tier**~~ — **done 1 Sep, and the framing was
   wrong.** At 5fps/150 calls the $1 cap is a ceiling of ~$3/Mtok output, which
   excludes every frontier model (sonnet-5 $3.33/video, gpt-5.1 $2.98,
   gemini-3.1-pro $3.81 — all at *standard* tier). Anthropic ruled out by the
   user as overpriced for vision. Mistral screened and rejected; kimi and glm
   blocked on the probe. **But frames are not fixed**, and on flex pricing
   `google/gemini-3.1-pro-preview` is **$0.85/video at 2fps** — the one
   untried route to a genuinely stronger model inside the cap. It rests on the
   §6 claim that better detections buy sparser sampling
   (`dt_max ≈ 1–2fps with perfect detections`). Probe its convention first;
   do **not** extrapolate from the flash-lite rows.
9. **Kalman options not yet explored**: render from the *filtered* state instead
   of raw observations (we compute it and discard it); a proper RTS backward
   smoother; adaptive process noise when a player accelerates; box size as part
   of the state rather than an EMA.
10. ~~**Collect the other four clips**~~ — **three collected 1 Sep** (§2).
    `football_cuts` has three verified hard cuts, so **D8 is finally testable**.
    One clip still wanted: a tight camera where players leave frame.
11. ~~**Two combination ablations on Luna**~~ — moot. Luna is no longer the
    model (D19); `--system` was rejected on Gemini anyway (+23% latency).

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
