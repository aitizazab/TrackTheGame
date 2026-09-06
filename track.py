"""
Stage 2: per-frame detections in, per-frame identities out.

    detections.json  --track--> tracks.json   (all 900 frames, every player
                                               carrying a stable id and number)

This is the layer the whole task is actually about. The VLM sees each frame in
isolation and has no idea the other frames exist; nothing it returns carries
identity. Everything that makes a marker stay attached to a person happens here,
in plain geometry, at a cost of milliseconds.

See docs/decisions.md. The short version of why it looks like this:

  D1  VLM perceives, geometry associates. The instructor permits classical CV for
      tracking, not detection, and it is the only arrangement that fits the
      latency budget — this file adds zero network round trips.

  D7  Labels are properties of the TRACK, not the frame. Jersey number and team
      are voted once over a track's whole lifetime and then painted onto every
      frame of it. A label decided once cannot flicker. This is only possible
      because we are offline: every tracker in the literature is online and
      cannot use the future. We can.

  D8  Camera cuts are checked BEFORE association, and re-anchored by number.

Borrowed, with sources:
  SORT        Kalman constant-velocity + Hungarian assignment. The skeleton.
  ByteTrack   two-stage association — match confident detections first, then let
              leftover tracks claim the hesitant ones. A hedged detection is
              usually an occluded player, not noise.
  OC-SORT     do not trust a long coast; correct velocity retroactively when a
              track reappears.
  BoT-SORT    compensate global camera motion before association. Ours is
              estimated from the detections themselves (median residual), never
              from pixels, which keeps it clear of the detection ban.
  DeepSORT    appearance re-ID. We keep the idea and drop the CNN: the jersey
              number IS our appearance feature, and the VLM already read it.

    uv run track.py outputs/detections/allstars_fr_eng__dev.json
    uv run track.py outputs/detections/allstars_fr_eng__dev.json --debug
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

OUT_DIR = Path("outputs/tracks")

# --- association geometry -------------------------------------------------
#
# Every threshold below is expressed in BODY HEIGHTS, not pixels and not
# fractions. That is what the bounding box bought us (D5): a player's box height
# is their apparent size, so a gate written in body-heights is automatically
# tighter for distant players and looser for near ones. One constant covers the
# whole frame instead of a compromise that is wrong at both ends.
#
# A sprinting human covers roughly 9 m/s and stands ~1.8 m, so about 5 body
# heights per second. 6 is that plus headroom for camera motion we failed to
# subtract.
# Was 8.0, calibrated on Luna alone. Measured across nine runs and six
# detectors: the gate must exceed roughly 3x a clip's own p90 frame-to-frame
# motion or tracks die and are re-born. At 8.0 that headroom is 4.3x for
# gemini-3.7-flash and 0.63-0.99x for four other detectors, and identity
# inflation tracks it exactly:
#
#     BH=8    g3.7flash 23 ids | luna 49 | qwen 125 | g3.1-lite 162 | g2.5-lite 201
#     BH=26   g3.7flash 23 ids | luna 38 | qwen  39 | g3.1-lite  39 | g2.5-lite  93
#
# 26 clears 3x on every recorded run. Above it inflation keeps falling but
# jersey-number purity starts to drop (luna 0.78 -> 0.67 at 40) — same-kit
# swaps that the inflation metric cannot see.
#
# THE UNDERLYING BUG, worth stating: "body heights" reads like a scene
# statistic and is a DETECTOR statistic. Median box height varies 2.6x between
# models on identical footage, while real player spacing varies only 2.4x
# across two clips and six models. Normalising by the box imported a per-model
# bias into a threshold meant to be scene-relative.
MAX_BODY_HEIGHTS_PER_SEC = 12.0
# The gate may not exceed this multiple of the physical speed limit, whatever
# the body-height scaling asks for. 4x is where the measurement is clean: at
# 5fps the worst marker jump is 0.0145 at 3-6x and 0.108 at 12x.
GATE_SPEED_HEADROOM = 4.0
# The gate may also not exceed this multiple of how far apart players actually
# stand in the clip. Measured on three runs; 1.5 halves the jumping frames and
# 1.0 is past the point where the gate stops covering real motion.
GATE_SPACING_MUL = 1.5
# Filled in by run() from the detections. A one-element list so gate_width can
# read it without threading a parameter through match().
CLIP_SPACING = [None]
# Above this second difference the path is genuinely curving and a centred
# smoother would push the marker forward along the curve rather than settle it.
# Detector jitter is ~0.003 per sample, so a curvature of 0.006 is already twice
# the noise it is meant to remove.
SMOOTH_MAX_CURVE = 0.006
# Set by --reconfirm. A lone sighting between two absences earns no marker.
RECONFIRM = [False]
GATE_FLOOR = 0.010          # fraction units; detector jitter on a static player
NUMBER_MATCH_BONUS = 0.55   # multiplies cost when jersey numbers agree
KIT_MISMATCH_PENALTY = 4.0  # multiplies cost across a team boundary; not a veto
# Apparent box height is a depth cue and is the only signal that survives a
# crossing, where screen separation goes to zero. Weight is deliberately mild:
# a 1.5x height disagreement costs 25% at 0.5, where the kit penalty costs 300%.
HEIGHT_MISMATCH_W = 0.5
# ---------------------------------------------------------- acceleration gate
#
# The existing gate limits SPEED. This one limits ACCELERATION, because the
# fly-out is not a fast player — it is a stationary-ish player whose reported
# position jumps and comes back. Speed alone cannot see that: 0.054 in 0.2s is
# inside the speed gate, which is why every fly-out sailed through.
#
# WHY ACCELERATION AND NOT A DIRECTION TEST. The user's objection to a
# direction-reversal rule is right: which way the jump goes is arbitrary, so a
# reversal test catches only the half that happen to double back. Acceleration
# is direction-agnostic — a jump ALONG the current velocity still produces a
# large change in velocity, so it is caught too, provided the change is big
# relative to the speed already held. The case it genuinely cannot see is a
# player already moving fast whose jump is small in proportion.
#
# DEPTH-NORMALISED, as required. The displacement is divided by the track's
# apparent box height, which is inversely proportional to distance from camera,
# so the unit is BODY HEIGHTS and a near player and a far one are judged alike.
# The x term carries the 16/9 correction: x is a fraction of frame WIDTH and h a
# fraction of frame HEIGHT, so comparing them raw understates x by 1.778 — the
# identical trap as the box-aspect guard in detect.py, and as HANDOFF §5's
# "box aspect looked fine in fraction space".
#
# THE THRESHOLD, and it is a NOISE threshold rather than a physical one.
# Measured over 23,477 consecutive observation triples from 5fps shipping-config
# runs (low-effort and 10fps runs excluded — acceleration is a second difference
# so its noise grows as 1/dt², and a 10fps run inflates the tail ~4x for reasons
# that have nothing to do with players):
#
#   p50 2.59   p90 6.96   p99 14.03   p99.5 18.08   p99.9 42.86   max 96.19
#
# There is NO GAP in this distribution — unlike the box-aspect guard, where real
# boxes stopped at 1.94 and corrupt ones began at 5.61. It is a smooth tail, and
# the reason is visible in the numbers: a footballer accelerates ~3 m/s², which
# at ~1.8 m per body height is 1.67 bh/s², and a sprinter peaks near 5.6. Our
# p90 is 6.96. So above roughly the 90th percentile this is not measuring player
# motion at all, it is measuring box-localisation noise differenced twice.
#
# The threshold is therefore calibrated against the fly-outs the USER identified
# by watching, which is the only ground truth available:
#
#   arm A  A·9   t+3.20s   a = 42.3   (p99.889)
#   arm B  B·15  t+9.00s   a = 49.8   (p99.945)
#   arm C  A·9   t+3.20s   a = 42.9   (p99.906)
#
# 30 sits 1.4x below the lowest of the three, and rejects 0.213% of triples —
# 50 of 23,477. A threshold of 50 catches NONE of them, so the margin matters
# more than the rejection rate here. Three events is a small sample: if a
# fly-out at 35 turns up, this number is why it was missed.
#
# Cost of a false positive is one frame of coasting, which the solver already
# prices; cost of a false negative is a visible fly-out.
# REFORMULATED 4 Sep, and the first version had a real defect.
#
# It gated on ACCELERATION, which divides by dt^2 - so a single missing frame
# doubles dt and makes the identical jump look FOUR TIMES gentler. The gate
# therefore weakened exactly when the tracker was most exposed, right after a
# dropped frame. Caught in the wild on football_cuts: white player `a` at
# t+2.00s, a 0.068 jump the model invented, which scored only 10.5 bh/s^2
# against a threshold of 30 purely because t+1.80s had never come back.
#
# The prediction ALREADY accounts for dt, so the residual from it needs no
# further time term. Measured over the same 23,477 triples, residual in body
# heights: p50 0.105, p90 0.277, p99 0.548, p99.9 1.687, max 3.848. The four
# fly-outs the user identified by watching all collapse into one narrow band
# once dt^2 is gone:
#
#   allstars A-9  t+3.20s   1.69      (was 42.3 bh/s^2)
#   cuts     B-15 t+9.00s   1.99      (was 49.8)
#   allstars A-9  t+3.20s   1.72      (was 42.9)
#   cuts     `a`  t+2.00s   1.69      (was 10.5 - MISSED by the old gate)
#
# 1.2 sits 1.4x below all four and rejects 0.164% of triples, against 0.213%
# for the acceleration form. Strictly better on both axes: catches one more
# real fly-out while refusing fewer honest detections.
#
# Still depth-normalised, and the 16/9 term is still mandatory - x is a fraction
# of frame WIDTH and h of frame HEIGHT.
#
# What it still cannot see: a drift assembled from many small steps, each under
# 1.2 on its own. The cuts case looked like that on screen but was not - it was
# one bad detection with an interpolated frame either side of it.
MAX_RESIDUAL_BH = 1.2
ACCEL_GATE_MIN_HITS = 3     # needs a velocity estimate worth predicting from
FRAME_ASPECT = 16 / 9
# What it costs to leave a track unmatched, or to call a detection a new player,
# expressed in gate widths so it stays commensurate with distance and scales
# with apparent size. Swept: 1.5 and 2.0 also fix the swap but merge two
# identities on basketball; 4.0 restores the bug exactly. 2.5 is the largest
# value that still fixes it, which keeps the most headroom for the D12 case a
# smaller value would break — a track with nothing same-kit in gate must still
# be allowed to take a flickered colour rather than die.
NO_MATCH_GATES = 2.5

# --- association instrumentation, OFF by default --------------------------
# --dump-costs FILE writes one JSON record per (track, candidate) pair that
# match() considers, carrying every term that went into the cost: the raw
# distance, the gate, the kit multiplier, the number bonus, the height term and
# the final cost, plus whether Hungarian took it. Nothing in this dict is read
# unless the flag is passed, so the shipped path is unchanged.
COST_LOG = {"on": False, "frame": None, "stage": "", "max_frame": 10 ** 9,
            "rows": []}
# Every association the acceleration gate refused, so the run summary can say
# how often it fired rather than leaving it as an invisible behaviour change.
ACCEL_REJECTS = []
# Off only for the A/B that justifies the gate. A list so the flag can be set
# from main() without threading a parameter through the whole call stack.
ACCEL_GATE_ON = [True]

HIGH_CONF = 0.50            # ByteTrack's split point
MIN_HITS = 3                # sightings before a track is real and drawable
# Two boxes overlapping more than this are the same player reported twice. Set
# well below the observed duplicate IoU (median 1.00) and well above what two
# genuinely adjacent players reach, since a player box is a tall thin sliver and
# neighbours barely touch.
DEDUPE_IOU = 0.45
# If deduplication removes more than this share of a frame's detections, the
# model was repeating itself rather than seeing a crowd.
DEGENERATE_RATIO = 0.40
MAX_COAST_S = 0.60          # kill a track unseen this long
COAST_DAMP = 0.72           # velocity decay per frame while unmatched
# The tracker had no concept of the frame edge. Death was decided on ONE
# condition — unseen for MAX_COAST_S — so a player running off the side and a
# player hidden behind a team-mate produced the identical signature (detections
# stop) and got identical treatment (extrapolate the last velocity for 0.6s).
# Coasting is right for occlusion and wrong for exit: the marker keeps sailing in
# the direction the player was running, which is the "ring sent flying" the user
# reported, and 42-46% of identities on the football clips ended within 6% of an
# edge. A track whose coasted centre crosses the boundary has left the picture;
# stop predicting and let re-entry start a clean track.
EDGE_MARGIN = 0.015         # a coasted centre this far outside [0,1] is gone
# Longer than this between two observations and the marker is NOT drawn across
# the gap. Short absences still interpolate — the player did not teleport — but
# a long one is where interpolation stops being a plausible guess and becomes a
# marker gliding through empty space to wherever the track resumed.
MAX_DRAW_GAP_S = 0.50
# A 9 m/s sprint across a ~68m pitch is ~0.13 fraction units/sec. 0.22 leaves
# headroom for camera motion and box jitter; past it, it is not a player.
MAX_PLAYER_SPEED = 0.22
# Round-trip threshold for reject_outliers, in BODY HEIGHTS of displacement.
# Measured over 28,959 consecutive player steps: p50 0.236, p90 0.610, p99
# 1.181. 0.35 is deliberately low because the three-point condition does the
# selecting - requiring both legs large AND the endpoints close fires on 15
# triples in 29,000 (0.053%), where a single-leg test at the same threshold
# would fire on roughly a quarter of them.
OUTLIER_JUMP_BH = 0.35
# A track must be SEEN this many times before it is drawn at all. MIN_HITS = 3
# governs whether a track is real; this governs whether it is shown. They were
# the same number, and 3-sample tracks are where the spurious ones live: A-4 on
# allstars is born at t+0.00s, wobbles, and dies at t+0.57s having never been a
# player. Measured across 669 tracks: 38 last exactly 3 samples (5.7%) and 42
# last 3 or fewer (6.3%), so this drops roughly one track in sixteen.
MIN_DRAW_SAMPLES = 4
# Marker easing, seconds. 0 disables it and restores the raw polyline, which is
# how the with/without comparison is rendered. 0.10 lags the target by about
# three frames at 30fps while a player accelerates, which is well under the
# marker's own radius, and removes the six-frame corner entirely.
SMOOTH_FOLLOW_S = 0.10
# Fade-out at the end of a drawn stretch, seconds. 0.30 = 9 frames at 30fps.
# Under ~0.15s reads as a pop and buys nothing over vanishing; over ~0.5s leaves
# a ring sitting on empty grass, which is the artefact being removed.
FADE_OUT_S = 0.30


def damped_follow(xs, ys, dt, smooth_time):
    """Critically damped follower over an already-interpolated path.

    The standard critically-damped spring solution, evaluated per frame. It is
    unconditionally stable and CANNOT overshoot, which is the whole reason for
    preferring it to a Catmull-Rom spline here: a spline passes exactly through
    every observation but overshoots on a sharp turn, and an overshooting
    smoother is a defect this project has already measured and guarded against
    once.

    Seeded at the first point so the marker does not slide in from the origin.
    """
    if smooth_time <= 0 or len(xs) < 2:
        return xs, ys
    omega = 2.0 / smooth_time
    ox = np.empty_like(xs)
    oy = np.empty_like(ys)
    px, py = float(xs[0]), float(ys[0])
    vx = vy = 0.0
    for i in range(len(xs)):
        tx, ty = float(xs[i]), float(ys[i])
        x = omega * dt
        # Pade approximation of exp(-x); standard in this formulation because it
        # stays stable for large dt without a transcendental call per frame.
        ex = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        cx, cy = px - tx, py - ty
        tmpx = (vx + omega * cx) * dt
        tmpy = (vy + omega * cy) * dt
        vx = (vx - omega * tmpx) * ex
        vy = (vy - omega * tmpy) * ex
        px = tx + (cx + tmpx) * ex
        py = ty + (cy + tmpy) * ex
        ox[i], oy[i] = px, py
    return ox, oy

# --- camera cut detection (D8) --------------------------------------------
PAN_SEARCH_FLOOR = 0.05     # search this far for a camera shift regardless of dt
# Cut detection, rebuilt 3 Sep on shot scale rather than association collapse.
# Thresholds sit between what a real cut produces and the worst a 30s no-cut
# control produces, with a wide margin on both sides:
#   scale jump   cuts 0.90 and 6.64        control never exceeds 0.19
#   count jump   cuts 7, 8 and 12          control never exceeds 3
CUT_SCALE_JUMP = 0.50       # |d median box height| / previous, as a ratio
CUT_COUNT_JUMP = 5          # absolute change in the number of players seen
CUT_MIN_TRACKS = 4          # never call a cut with fewer live tracks than this
# COUNT RATIO, added 4 Sep, and it is the test that finds the cut the other two
# cannot. The 18.3s cut on football_cuts goes from a 1-player goalkeeper
# close-up to a 4-player scene: the ABSOLUTE change is 3, below CUT_COUNT_JUMP,
# and CUT_MIN_TRACKS = 4 was additionally suppressing the whole boundary as
# "thin evidence". But 1 -> 4 is a FOURFOLD change, which is not thin at all -
# the absolute form simply cannot see a big change between small numbers.
#
# Measured on every sampled boundary in the project. True cuts: 4.0, 5.0, 8.0,
# 9.0. Cut-free clips - allstars, basketball x2, amateur, 595 boundaries - top
# out at 1.50. That is an empty band from 1.5 to 4.0, so 2.5 is placed in a gap
# the data has rather than tuned to a target, the same way the box-aspect guard
# was.
#
# Validated: this rule plus the scale and kit tests catches 4 of the 5 real cuts
# with ZERO false positives across all five clips.
#
# The fifth cut, at 8.8s, is INVISIBLE in the detection stream and no threshold
# will find it: 18 players either side, median box height 0.075 -> 0.080, kit
# distribution identical, and the model's own scene sentence MORE similar than a
# typical non-cut boundary. It is a cut between two near-identical wide
# broadcast angles. Detecting it needs pixels, which D8 already measured ffmpeg
# to be bad at on football. Stated as a limitation rather than chased.
CUT_COUNT_RATIO = 2.5
# Kit-mix change, L1 over the per-frame colour distribution. True cuts 0.67-1.78
# against a cut-free maximum of 0.56, so 0.9 clears both sides.
CUT_KIT_L1 = 0.90
# A real cut is one event, but the frame after it often still disagrees with the
# frame before it, so the tests fire twice. Observed: 642 AND 648 on the 21.6s
# cut, 0.2s apart, retiring every track twice and killing the tracks the first
# retirement had just created. A cut cannot be followed by another cut inside a
# fifth of a second in any footage we would use.
#
# KEEP THE STRONGEST, NOT THE FIRST - corrected 5 Sep after the first version
# made things worse. Suppressing later firings kept 642 and discarded 648, but
# 648 is the REAL boundary: tracks were retired 0.2s early, re-born in the
# 642-648 window, and then interpolated straight across the actual cut, which
# is the artefact the cut detector exists to prevent. The user spotted it as a
# ring gliding through the 21.6s cut.
#
# Evidence strength is the count ratio, which is the test with the widest
# separation from non-cut boundaries (true cuts 4.0-9.0, cut-free max 1.50).
CUT_DEBOUNCE_S = 0.50

# --- ball -----------------------------------------------------------------
# The ball is not a person and does not obey body-height scaling: a struck ball
# moves 30 m/s and is a few pixels across. Flat generous gate instead.
BALL_GATE_PER_SEC = 1.2     # fraction units per second a real ball can manage
BALL_JUMP_FRAC = 0.10       # a "leap" for round-trip purposes
# DEPTH-NORMALISED FORMS, added 5 Sep. The two constants above are flat frame
# fractions, justified by a comment saying the ball "does not obey body-height
# scaling" because it is a projectile rather than a runner. That conflates two
# different things. A ball's REAL-WORLD speed is indeed not body-scaled - but
# its IMAGE displacement scales with depth exactly like everything else, and the
# gate is applied in image units.
#
# Measured median player height per frame, which is a 1/depth proxy:
#
#   allstars 0.071-0.117 (2.0x)   basketball 0.193-0.248 (1.7x)
#   amateur  0.077-0.100 (1.9x)   cuts       0.068-0.645 (15.9x)
#
# So one flat number is asked to serve a ball that is up to 16x closer to the
# camera in one part of a clip than another, and the median scale differs 2.6x
# between basketball and the football clips. Too tight for a near ball, too
# loose for a far one - the same error the player gate made before D12.
#
# RE-EXPRESSED, NOT RETUNED. Both are the old constant divided by the allstars
# median player height (0.084), so behaviour on the reference clip is unchanged
# and only the scaling with depth is new.
BALL_JUMP_PH = 1.19             # 0.10 / 0.084, in player heights
BALL_GATE_PH_PER_SEC = 14.3     # 1.2  / 0.084, in player heights per second
# ...and the flat constants stay on as an absolute CEILING, so normalising can
# only ever TIGHTEN the gate, never loosen it. This is the same two-limit design
# gate_width already uses for players: body-height scaling governs the small
# end, a flat cap governs the large one.
#
# It is here because the first version had no ceiling and measurably regressed
# basketball - median player height 0.217 against allstars' 0.084 made the gate
# 2.6x looser and ball rejections fell 9 -> 2 on the clip whose decoys are the
# worst in the set. A principled change is not automatically an improvement;
# the direction it moves each clip has to be checked.
#
# With the cap: allstars unchanged (14.3 x 0.084 = 1.20 = the cap), basketball
# unchanged (capped), and a DISTANT ball - scale 0.05 - gets 0.715 instead of
# 1.2, which is the tightening that was the whole point.
# Below this many players the frame has no usable scale, so fall back to the
# flat constants rather than normalise by noise.
BALL_SCALE_MIN_PLAYERS = 4
# The round trip and the speed gate run alternately until neither removes
# anything. Capped rather than unbounded: each pass makes survivors look more
# isolated, and an over-eager ball filter has already cost this project real
# detections once (see the retired static-cluster note below).
BALL_FILTER_PASSES = 4
# The test cannot tell a painted mark from a ball lying still on the grass while
# the camera pans across it — both sit at a fixed point on the pitch. What
# separates them is persistence, so the bar is "seen at the same field position
# this many times inside a two-second window".
# A painted mark on the turf is squashed vertically by the oblique camera; a
# ball is a sphere and is not. Measured on this clip, box WIDTH is identical
# between the two populations (0.0080 vs 0.0075) and HEIGHT is nearly 2x apart
# (0.0120 vs 0.0065). Expressed as a fraction of the clip's own median ball
# height so it survives a change of shot scale.
BALL_FLAT_H_FRAC = 0.65
# Back to 5. It was dropped to 3 to catch a decoy that kept slipping through,
# but the real problem was that clustering happened in SCREEN coordinates, so a
# panning camera scattered one decoy into several small clusters. Lowering the
# bar just started rejecting real ball detections as well. With clustering done
# in the stabilised frame the decoy forms one large cluster again and the bar can
# go back up, which is where the false positives came from.
# How long a ball may go unseen and still have a straight line drawn across the
# absence. Beyond it the marker disappears rather than inventing a path.
# Measured on gemini-3.7-flash 30s/1080 — ball frames drawn:
#     0.25s -> 640    0.50s -> 771    1.00s -> 817    2.00s -> 817
# 1.0 gains 46 frames over 0.5 and 2.0 gains nothing further, so the real gaps
# in this footage are all under a second and 1.0 is the natural plateau.
BALL_MAX_GAP_S = 1.00
# Hard floor on the ball's reported confidence, OFF by default (--ball-min-conf).
# A 0.70 floor was measured and rejected on Luna: 20 of 37 decoys caught against
# 19 of 252 good detections lost, roughly one-for-one. That was a different model
# on a different clip, and the decoys Gemini 3.7 Flash produces sit at 0.70-0.80
# against 0.90-0.95 for real balls, so the separation may be cleaner here.
# Left as a flag rather than a default so the claim gets tested, not assumed.
BALL_MIN_CONF = 0.0
# TIGHTENED 1.6 -> 0.4 on 5 Sep. This is distance from the nearest point ON the
# player's box, in units of that player's box height, so 1.6 allowed the ball to
# sit 1.6 player-heights BEYOND the box edge - roughly 2.9m for a footballer -
# and still count as possession.
#
# Swept across all four clips (on-ball frames at each radius):
#
#   radius   allstars  basketball  amateur  cuts
#     1.6      730        859        642     602
#     0.8      666        859        628     569
#     0.4      586        857        576     498
#     0.0      216        723        309     221
#
# Halving it does almost nothing, because the radius is rarely what binds: the
# median nearest player is already 0.14-0.19 heights away. 0.4 is chosen on the
# BASKETBALL column - it moves that clip 859 -> 857, i.e. it leaves untouched
# the one the user judged correct, while trimming the three judged wrong by
# 10-20%. A threshold that only moves the cases called wrong is the right shape.
# 0.0 (containment only) would gut football, which does have real possession.
ON_BALL_RADIUS_BH = 0.4     # a player is "on the ball" within this many heights
# A possession spell shorter than this is the ball passing by, not possession.
MIN_POSSESSION_S = 0.40
# --- Kalman size + adaptive noise -----------------------------------------
# How fast apparent size is allowed to drift, as a multiple of the position
# process noise. Small: depth changes slowly compared with lateral motion.
SIZE_PROCESS = 0.35
# Measurement noise on w and h, as a fraction of h. Deliberately larger than
# the 0.06 used for position - the model re-estimates the box every frame and
# height was measured wobbling 0.057 -> 0.043 -> 0.057 on a stationary player.
SIZE_MEAS_NOISE = 0.18
# Adaptive process noise: how much a manoeuvring track may inflate Q, and how
# strongly its recent innovation drives that. Bounded so a run of bad
# detections cannot open the gate indefinitely.
# MEASURED AND DISABLED, 5 Sep. Kalman option 3 - inflate Q when a track keeps
# surprising its own prediction - is theoretically the right answer to the
# "box lags under acceleration" defect, and it measurably made things worse:
# isolated from the size change, it added one spurious identity to basketball
# (18 -> 19) and one to football_cuts (49 -> 50) and improved nothing on any of
# the five clips. Loosening Q loosens association, and the lag it buys back
# costs more than it saves. Kept at 0 rather than deleted so the ablation is
# reproducible; set to 2.5 to re-enable.
MANOEUVRE_GAIN = 0.0
MANOEUVRE_MAX = 3.0
# DEPTH CHECK ON POSSESSION, added 4 Sep.
#
# Possession was decided purely in 2D - containment, else distance to the box -
# so a ball high in the air projects onto players standing much further up the
# pitch, and one of them is credited. The user watched exactly that on the
# amateur clip: the ball well above the ground and a player nowhere near it
# holding the marker.
#
# Apparent size is the depth cue, and both objects already report it. A 0.22m
# ball against a 1.8m player subtends ~0.12 of their height AT THE SAME DEPTH.
# Measured ball_h / median player_h on real detections:
#
#   amateur    p10 0.116  p50 0.146  p90 0.214  max 0.429
#   basketball p10 0.118  p50 0.152  p90 0.177  max 0.225
#   cuts       p10 0.133  p50 0.158  p90 0.188  max 0.231
#
# Centred near 0.15 rather than 0.12 because the model boxes the ball loosely,
# and tight enough across three clips and two sports to be worth trusting. The
# ratio scales as (player depth / ball depth), so 0.35 means the claimed player
# is ~2.4x further away than the ball is - which is not possession, it is a
# coincidence of projection. Deliberately wide: this is meant to remove the
# absurd cases, not to adjudicate close ones.
BALL_DEPTH_RATIO_MAX = 0.35
BALL_DEPTH_RATIO_MIN = 0.05
# GROUND PLANE. The ratio test above turned out to be a near no-op - it changed
# 2 possession frames of 790 - because a ball lofted high but at the SAME depth
# has a perfectly normal apparent size. Size sees depth; it cannot see height
# above the pitch, which is the thing that was actually wrong.
#
# The players themselves give us the pitch. Perspective makes a standing
# player's foot height a near-linear function of their apparent size, so a
# least-squares fit of foot_y against box_h over the players in one frame IS the
# ground plane. Measured R^2 (p50) per clip:
#
#   amateur 0.96   allstars 0.82   cuts 0.79   basketball 0.61
#
# 0.96 on the clip where the user saw the defect. Basketball is weakest, which
# is expected and harmless: a tight camera has little perspective spread, and
# the ball there is genuinely held at chest height rather than lofted.
#
# Given the ball's own size we know its depth, so we know how tall a player at
# the ball's distance would be, so we know where the pitch is under it. A ball
# far ABOVE that line is in flight and nobody is holding it.
GROUND_FIT_MIN_PLAYERS = 6      # below this the fit is not worth trusting
GROUND_FIT_MIN_R2 = 0.55        # nor is it if the frame does not fit a plane
# How far above the predicted ground line, in ball-implied body heights, before
# possession is refused outright. 1.5 is roughly head height, so a header still
# counts as possession and a goal kick does not.
BALL_AIRBORNE_BH = 1.5
ON_BALL_SMOOTH_S = 0.40     # RETIRED: the old symmetric majority-vote window
# How long a challenger must be the per-frame pick before possession transfers.
# Asymmetric on purpose: keeping the ball needs no evidence, taking it does.
#
# Swept on both clips. Every genuine turnover is delayed by exactly this much,
# so it is a straight trade of lag against flicker:
#
#             basketball spells / median      football spells / median
#   0.0s        55 / 0.20s   (unusable)         33 / 0.33s
#   0.2s        29 / 0.60s                      25 / 0.60s
#   0.3s        16 / 1.55s   <- knee            20 / 0.82s
#   0.4s        13 / 2.53s                      17 / 1.07s
#   0.6s        11 / 2.70s                      14 / 1.23s   (visibly laggy)
#
# 0.3 is where basketball collapses from 29 spells to 16 — the flicker is gone —
# while football keeps turnovers at 0.82s, close to the real thing. Past 0.3 the
# curves flatten and all that is bought is lag, which is what showed on the
# football render at 0.6.
ON_BALL_STICK_S = 0.30


def foot(d: dict) -> tuple:
    """Where the player meets the ground: bottom-centre of the box.

    Derived rather than asked for. One less number for the model to get wrong,
    and the point the marker is drawn at.
    """
    return d["x"] + d["w"] / 2.0, d["y"] + d["h"]


class Kalman:
    """Constant-velocity filter on the foot point. State [x, y, vx, vy].

    Constant velocity is a first-order approximation and it is wrong the moment
    a player changes direction. It survives anyway because dt is small (D9: we
    sample at 10-15fps precisely so that it is) and because a wrong prediction
    only has to be closer to the right player than to any other one.
    """

    def __init__(self, pos, h, w=None):
        # STATE IS NOW SIX-DIMENSIONAL: [x, y, vx, vy, w, h].
        #
        # Box size used to be an EMA outside the filter (0.7/0.3), which gave a
        # smoothed number with no uncertainty attached and no way to tell a real
        # size change from a bad measurement. That mattered more than it looks:
        # almost everything added to this tracker divides by h - the residual
        # gate, the possession radius, the ball's depth-normalised gates, the
        # ground-plane fit - so noise in h propagates into all of them at once.
        # A concrete case: the white 14/24 swap on football_cuts, where the
        # depth cue was real (0.057 vs 0.069) but the raw height wobbled
        # 0.057 -> 0.043 -> 0.057 across three frames and buried it.
        #
        # Size is modelled as constant plus noise, not as having a velocity: a
        # player's apparent size drifts with depth, it does not accelerate.
        self.x = np.array([pos[0], pos[1], 0.0, 0.0,
                           (w if w is not None else h * 0.28), h], dtype=float)
        self.P = np.diag([1e-3, 1e-3, 1e-2, 1e-2, 1e-4, 1e-4])
        # EMA of the recent normalised innovation, for adaptive process noise.
        self.manoeuvre = 0.0

    @property
    def h(self):
        return float(self.x[5])

    @property
    def w(self):
        return float(self.x[4])

    def predict(self, dt):
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        self.x = F @ self.x
        # Process noise scales with size: a near player's real-world wobble is
        # a larger fraction of the frame than a distant player's.
        q = (self.h * dt) ** 2
        # ADAPTIVE. A constant-velocity model is wrong exactly when a player is
        # accelerating, and a fixed Q makes the filter equally confident either
        # way - so it lags on a burst and is guarded by a gate that must then be
        # loose enough to tolerate the lag everywhere. `manoeuvre` is an EMA of
        # the recent innovation measured in body heights, so a track that keeps
        # surprising its own prediction is granted more process noise until it
        # settles. Bounded so a run of bad detections cannot open the gate up
        # indefinitely, which is how an adaptive scheme turns into no scheme.
        boost = 1.0 + min(MANOEUVRE_MAX, MANOEUVRE_GAIN * self.manoeuvre)
        qp, qv = q * 0.25 * boost, q * 4.0 * boost
        qs = (self.h * dt * SIZE_PROCESS) ** 2
        Q = np.diag([qp, qp, qv, qv, qs, qs])
        self.P = F @ self.P @ F.T + Q
        return self.x[:2].copy()

    def update(self, pos, conf, h, w=None):
        H = np.zeros((4, 6))
        H[0, 0] = H[1, 1] = 1.0      # x, y
        H[2, 4] = H[3, 5] = 1.0      # w, h
        # A hesitant detection is trusted less. conf 1.0 -> tight, 0.2 -> loose.
        #
        # The coefficient was 0.35, which put the measurement standard deviation
        # at ~0.029 fraction units — THREE TIMES the median distance a player
        # actually moves between samples (0.0092, measured). The filter therefore
        # believed its own drifting prediction over the detection in front of it.
        # 0.06 puts it near real detector jitter instead of near a player's whole
        # stride.
        r = (h * 0.06 / max(conf, 0.15)) ** 2
        # Size is measured far more noisily than position - the model re-draws
        # the box from scratch every frame - so it gets its own, looser R. This
        # is what replaces the old 0.7/0.3 EMA: same "size changes slowly"
        # intent, but expressed as a measurement the filter can weigh rather
        # than a blend it must accept.
        rs = (h * SIZE_MEAS_NOISE) ** 2
        R = np.diag([r, r, rs, rs])
        z = np.array([pos[0], pos[1],
                      w if w is not None else self.w, h], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        # Innovation in body heights, EMA'd, feeds the adaptive process noise.
        innov = float(np.hypot(y[0] * FRAME_ASPECT, y[1])) / max(self.h, 0.02)
        self.manoeuvre = 0.7 * self.manoeuvre + 0.3 * innov

    def coast(self):
        """Bleed velocity off a track that found nobody this frame.

        A constant-velocity model given no corrections sails on forever at
        whatever speed it last inferred. After half a second of that, the
        prediction is somewhere the player never went — and a detection landing
        near the *drifted prediction* is far from the last real observation, so
        the re-match is geometrically wrong and the interpolated marker slides
        across the pitch to reach it. Decaying toward a standstill keeps a lost
        track near where it was actually last seen.
        """
        self.x[2] *= COAST_DAMP
        self.x[3] *= COAST_DAMP

    def retro_correct(self, pos, gap_s):
        """OC-SORT: after a long coast the velocity estimate has drifted, because
        it was integrated for many steps with nothing to correct it. Rebuild it
        from where the track actually reappeared instead of trusting the drift."""
        if gap_s > 1e-6:
            self.x[2] = (pos[0] - self.x[0]) / gap_s
            self.x[3] = (pos[1] - self.x[1]) / gap_s
        self.x[0], self.x[1] = pos


class Track:
    _next = 1

    @property
    def h(self):
        return self.kf.h

    @property
    def w(self):
        return self.kf.w

    def __init__(self, det, frame, t):
        self.id = Track._next
        Track._next += 1
        self.kf = Kalman(foot(det), det["h"], det["w"])
        self.hits = 1
        self.last_t = t
        self.last_frame = frame
        self.kit_votes = Counter()
        self.num_votes = defaultdict(float)   # number -> summed confidence
        self.num_counts = Counter()           # number -> how many times seen
        self.obs = []                          # (frame, x, y, w, h) for interpolation
        self.absorb(det, frame, t)

    def absorb(self, det, frame, t):
        self.kit_votes[det["kit"].strip().lower()] += 1
        if det["num"] is not None:
            self.num_votes[int(det["num"])] += float(det["conf"])
            self.num_counts[int(det["num"])] += 1
        # w and h now come from the filter (see Kalman.__init__), so there is
        # no second smoother here to disagree with it. Every consumer of tr.h -
        # the gate, the residual test, the height-mismatch cost, possession -
        # reads the same filtered estimate.
        fx, fy = foot(det)
        self.obs.append((frame, fx, fy, self.w, self.h))
        self.last_t = t
        self.last_frame = frame

    @property
    def kit(self):
        return self.kit_votes.most_common(1)[0][0] if self.kit_votes else None

    @property
    def number(self):
        """Modal jersey number, weighted by confidence.

        Requires total weight >= 1.0 so that one blurry glance at a 7 does not
        brand a track forever. Below that the renderer falls back to the track's
        own id, which per the spec is an acceptable arbitrary-but-stable label.
        """
        if not self.num_votes:
            return None
        n, w = max(self.num_votes.items(), key=lambda kv: kv[1])
        # Two independent sightings, not one confident glance. A single read at
        # conf 0.98 used to clear the old weight-only bar on its own, which is
        # how a 13 became a 93: one misread branded the track permanently. The
        # count requirement is the part that matters — a misread rarely repeats,
        # because the next frame shows the shirt at a different angle.
        return n if (w >= 1.2 and self.num_counts[n] >= 2) else None


def scene_jaccard(a: str, b: str) -> float:
    """Word overlap between two scene sentences.

    `scene` exists in the schema as a reasoning scratchpad — somewhere for the
    model to think before committing to coordinates. It turns out to double as a
    shot-change signal for free, because a cut changes the framing description.
    """
    if not a or not b:
        return 1.0
    sa = {w for w in a.lower().split() if len(w) > 3}
    sb = {w for w in b.lower().split() if len(w) > 3}
    if not sa or not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def estimate_global_motion(preds, dets, max_shift):
    """Recover the camera's translation between two frames from the detections.

    A broadcast camera pans constantly, and under a pan EVERY predicted position
    is wrong by the same vector. The first version estimated that vector from the
    residuals of matches it had already made, and applied it to the next frame —
    which lags by one frame and collapses entirely when the pan is fast enough to
    break the matches it depends on. That circularity is why panning shredded the
    tracks: association failed, so there were no residuals, so no compensation was
    estimated, so association failed harder.

    This estimates the shift BEFORE associating anything, using a crude
    vote: every plausible (prediction, detection) pairing casts one vote for the
    displacement it implies, votes are binned, and the winning bin is the shift
    that the largest number of pairings agree on. Wrong pairings scatter their
    votes; correct ones pile into the same bin. A Hough transform in miniature.

    It also gives us the signal that separates a PAN from a CUT: under a pan the
    votes concentrate, under a cut they scatter.

    Returns (dx, dy, support) where support is the share of tracks that agree.
    """
    if not preds or not dets:
        return 0.0, 0.0, 0.0
    bin_w = max_shift / 12.0
    votes = defaultdict(list)
    for px, py in preds:
        for d in dets:
            fx, fy = foot(d)
            dx, dy = fx - px, fy - py
            if abs(dx) > max_shift or abs(dy) > max_shift:
                continue
            votes[(round(dx / bin_w), round(dy / bin_w))].append((dx, dy))
    if not votes:
        return 0.0, 0.0, 0.0
    best = max(votes.values(), key=len)
    arr = np.array(best)
    return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])),
            len(best) / len(preds))


def gate_width(h, dt):
    """How far a track may reach for its next detection.

    TWO limits, and the tighter wins. They express different things and the file
    used to contain only the first, which let them disagree by an order of
    magnitude:

      body-height scaling  handles PERSPECTIVE. A near player is larger and
                           crosses more of the frame per second than a distant
                           one, so the reach has to scale with apparent size.
      absolute speed cap   handles PHYSICS. No player, near or far, exceeds
                           MAX_PLAYER_SPEED. Body-height scaling has no idea
                           this limit exists and will happily grant a reach of
                           half the frame width to a large box.

    Measured at 5fps: at BH=26 the gate was 0.546 fraction units, 12.4x what
    MAX_PLAYER_SPEED permits, and the worst marker jump went from 0.0145 to
    0.1082 — markers visibly flying to other players. Identity COUNT barely
    moved (28 vs 31), which is why a count-based check missed it entirely: a
    track matching the wrong player keeps the count and loses the person.

    The cap bites for large boxes and the scaling governs small ones, which is
    the division of labour each was actually good at.
    """
    g = min(MAX_BODY_HEIGHTS_PER_SEC * max(h, 0.02),
            GATE_SPEED_HEADROOM * MAX_PLAYER_SPEED) * dt + GATE_FLOOR
    # THIRD limit: how far apart the players actually are in THIS clip.
    #
    # The first two are about one player in isolation — how big they look, how
    # fast a human runs. Neither knows whether the gate encloses one candidate
    # or six, and that is what decides whether Hungarian picks the right one.
    #
    # Measured: the crowded middle of the clip packs players 18% closer (spacing
    # p25 0.042 against 0.051 in the open first third) while the gate stays
    # fixed, pushing gate/spacing from 3.6x to 4.4x — into the band where swaps
    # appear. Capping at 1.5x spacing halved the frames that jump (1.4% -> 0.7%)
    # and cut worst-case 0.0183 -> 0.0142. At 1.0x it breaks: the gate falls
    # below real motion and tracks die instead of matching.
    #
    # Spacing is a SCENE statistic — it varied 2.4x across two clips and six
    # models where box height varied 2.6x between models on identical footage.
    # That is the kind of number that transports.
    if CLIP_SPACING[0]:
        g = min(g, GATE_SPACING_MUL * CLIP_SPACING[0])
    return g


def dedupe(players):
    """Drop detections that are duplicates of a more confident one.

    The ghost markers — ellipses sitting on empty grass for half a second — are
    not hallucinations and not tracker drift. The model emits the SAME PLAYER
    TWICE in the same frame: 211 overlapping pairs across 76 of 292 frames, and
    the median IoU of those pairs is **1.00**, i.e. byte-identical boxes. In 78
    of them the two copies are even given different kit colours, so one player is
    reported as both white and blue at once.

    The duplicate spawns its own track, drifts off on its own association
    mistakes, and dies a few frames later. Nothing downstream can distinguish it
    afterwards — measured, short-lived tracks and long ones are indistinguishable
    by confidence (0.91 vs 0.94) and identical in box height. So it has to be
    removed here, before it becomes a track.

    Standard non-maximum suppression, on model output rather than pixels: sort by
    confidence, keep a box unless it overlaps something already kept.
    """
    out = []
    for p in sorted(players, key=lambda q: -q.get("conf", 0)):
        x2, y2 = p["x"] + p["w"], p["y"] + p["h"]
        dup = False
        for q in out:
            qx2, qy2 = q["x"] + q["w"], q["y"] + q["h"]
            iw = max(0.0, min(x2, qx2) - max(p["x"], q["x"]))
            ih = max(0.0, min(y2, qy2) - max(p["y"], q["y"]))
            inter = iw * ih
            union = p["w"] * p["h"] + q["w"] * q["h"] - inter
            if union > 0 and inter / union > DEDUPE_IOU:
                dup = True
                break
        if not dup:
            out.append(p)
    return out


def letter_label(n: int) -> str:
    """0 -> 'I', 1 -> 'II', 3 -> 'IV'. ROMAN NUMERALS, not letters.

    Changed 6 Sep. Invented labels were spreadsheet-column letters - a, b, ...
    z, aa - drawn beside real jersey numbers. Three problems, and the third is
    the one that decided it:

      - `A-h` reads as a typo, or as a corrupted number, rather than as a
        deliberate statement that we could not read this shirt.
      - the separator was a middle dot, which the Windows console cannot print
        and which has already been mistaken for data corruption once.
      - lowercase letters have descenders and wildly uneven widths, so a column
        of them jitters as tracks come and go.

    Roman numerals are unmistakably NOT jersey numbers - no footballer wears
    VII - so they need no legend, no opacity trick and no colour change, which
    is the same reasoning that chose letters originally but executed better.
    They are all caps, all straight strokes, and legible at 9-21px.

    Falls back to the old letters above 3999, which cannot occur (a clip has
    tens of tracks, not thousands) but should not raise if it somehow does.
    """
    n = int(n) + 1                      # labels are 1-based: 0 -> I
    if not 0 < n < 4000:
        s = ""
        m = int(n) - 1
        while True:
            s = chr(ord("a") + m % 26) + s
            m = m // 26 - 1
            if m < 0:
                return s
    out = []
    for value, sym in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                       (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                       (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def drop_unconfirmed(o, step):
    """Discard a sighting that stands alone between two gaps.

    Luna's per-frame player count churns about 15% — the same borderline players
    drop in and out of the list frame to frame while everyone's position stays
    put. A track kept alive across those misses is drawn over empty grass, which
    is the "rings without players" in the opening seconds.

    The rule: a sighting only earns a marker if it is adjacent to another one.
    Seen once between two absences, it is more likely the detector twitching
    than a player appearing for a fifth of a second and vanishing.

    This keeps the TRACK alive — identity survives, the marker just stops being
    drawn during the absence. That is the difference from shortening the coast,
    which throws the identity away as well.
    """
    if len(o) < 3:
        return o
    lim = step * 1.5
    keep = []
    for i, p in enumerate(o):
        prev_ok = i > 0 and (p[0] - o[i-1][0]) <= lim
        next_ok = i < len(o)-1 and (o[i+1][0] - p[0]) <= lim
        if prev_ok or next_ok:
            keep.append(p)
    return keep


def smooth_observations(o, src_fps, nominal_dt):
    """Zero-phase smoothing over a finished track's observations.

    The Kalman filter is CAUSAL: at frame N it knows frames 1..N and nothing
    else, so the best it can do about a jittery measurement is split the
    difference with its own prediction. We are offline and have the whole track,
    so we can use the future as well as the past — which is what a smoother is.

    Why this matters: a stationary player's box wobbles ~0.003 between frames
    because the model re-estimates it from scratch every time, and the renderer
    interpolates between the RAW observations, passing that wobble straight
    through to the marker. The Kalman state we spent all that effort computing
    was being discarded at render time.

    A centred binomial kernel [0.25, 0.5, 0.25]. Centred means ZERO LAG — a
    causal average would delay every marker behind its player, which is exactly
    the artifact the camera-pan lag already produces. Three points is
    deliberately mild: it halves independent jitter while barely rounding a real
    direction change, and a wider window would start cutting corners off runs.

    Not applied across a long gap, where the neighbours are not evidence about
    this point at all.
    """
    if len(o) < 3:
        return o
    max_gap = 2.5 * nominal_dt * src_fps
    out = [o[0]]
    for a, b, c in zip(o, o[1:], o[2:]):
        if (b[0] - a[0]) > max_gap or (c[0] - b[0]) > max_gap:
            out.append(b)
            continue
        # SECOND DIFFERENCE: how curved the path is here. On a straight run it
        # is ~0 and any deviation is jitter, which is what we want to remove.
        # On a curve it is the real acceleration.
        #
        # A centred average OVERSHOOTS a curve, and forward along it. Take
        # a=0, b=1, c=3: the smoothed value is 0.25(0) + 0.5(1) + 0.25(3) = 1.25,
        # ahead of the true 1. Under a camera pan that speeds up, every marker
        # is accelerating together — so every marker gets pushed AHEAD of its
        # player, exactly the artifact reported. Smoothing did not create the
        # pan lag, it inverted and amplified it.
        #
        # So smooth only where the path is straight enough that the correction
        # is smaller than the jitter it removes.
        d2x = c[1] - 2 * b[1] + a[1]
        d2y = c[2] - 2 * b[2] + a[2]
        if (d2x * d2x + d2y * d2y) ** 0.5 > SMOOTH_MAX_CURVE:
            out.append(b)
            continue
        out.append((b[0],
                    0.25 * a[1] + 0.5 * b[1] + 0.25 * c[1],
                    0.25 * a[2] + 0.5 * b[2] + 0.25 * c[2],
                    b[3], b[4]))
    out.append(o[-1])
    return out


def reject_outliers(o, src_fps):
    """Drop observations a player could not physically have reached and left.

    Exactly the round-trip test used on the ball, applied to players — the same
    signature turns out to mean the same thing for both. If getting to point B
    and away from it both demand superhuman speed, but going straight from A to
    C does not, then B was an association error and the track never went there.

    Deleting the point keeps the path CONTINUOUS, which splitting the track did
    not: interpolation simply spans the hole, so the marker neither teleports
    nor blinks out.

    REWRITTEN 5 Sep - it was expressed in SPEED, and speed divides by dt.

    Two fly-outs the user found by watching survived it, and both for the same
    reason. B22 on allstars at t+4.20s goes 0.013 -> 0.074 -> 0.039, an obvious
    out-and-back. But t+4.40s was never returned by the model, so the RETURN leg
    spans 0.4s instead of 0.2s and its speed halves: 0.0875 frac/s against a
    0.22 limit. The displacement was damning and the speed was unremarkable.

    This is the third place today the same dt bug appeared - the association
    gate divided by dt^2, the ball speed gate by dt, and this by dt. A dropped
    frame silently disarms every test built on a rate.

    Now expressed in DISPLACEMENT, body heights, with the ball's ratio form:
    b is a fly-out if getting to it and away from it are both large moves, while
    going straight from a to c is not. Depth-normalised, so a near player who
    genuinely covers more pixels is judged on the same scale as a distant one,
    and the 16/9 term is there because x is a fraction of WIDTH and h of HEIGHT.

    Measured over 28,959 consecutive player steps: p50 0.236 bh, p90 0.610,
    p99 1.181. At OUTLIER_JUMP_BH = 0.35 the full three-point condition fires on
    15 triples in 29k - 0.053% - because requiring BOTH legs to be large AND the
    endpoints to be close is far more selective than either test alone.
    The two known cases: A-4 steps 0.41/0.47, B22 steps 0.86/0.47.

    Observation tuples are (frame, x, y, w, h), so the height is index 4.
    """
    if len(o) < 3:
        return o
    keep = [o[0]]
    i = 1
    while i < len(o) - 1:
        a, b, c = keep[-1], o[i], o[i + 1]

        def bh(p, q):
            """Displacement p->q in body heights, depth-normalised."""
            h = max((p[4] + q[4]) / 2, 0.02)
            return float(np.hypot((q[1] - p[1]) * FRAME_ASPECT,
                                  q[2] - p[2])) / h

        d_ab, d_bc, d_ac = bh(a, b), bh(b, c), bh(a, c)
        if (d_ab > OUTLIER_JUMP_BH and d_bc > OUTLIER_JUMP_BH
                and d_ac < 0.5 * d_ab):
            i += 1                      # b was never there
            continue
        keep.append(b)
        i += 1
    # The last point has no successor, so the three-point test cannot see it —
    # and the loop appended it unconditionally, which let a bad final
    # association drag the marker off on the way out. Judge it against the
    # trailing path instead, and drop it if it is unreachable.
    last, prev = o[-1], keep[-1]
    h = max((prev[4] + last[4]) / 2, 0.02)
    reach = float(np.hypot((last[1] - prev[1]) * FRAME_ASPECT,
                           last[2] - prev[2])) / h
    # Judged in body heights for the same reason as the loop above: the old
    # speed form let a long gap disguise a large final jump. 2x the round-trip
    # threshold, because there is no successor to confirm the return - a lone
    # last point gets more benefit of the doubt than one caught mid-trip.
    if reach <= 2.0 * OUTLIER_JUMP_BH:
        keep.append(last)
    return keep


def match(tracks, preds, dets, dt, cam, team_of):
    """Hungarian assignment of predicted track positions to detections.

    Returns (pairs, unmatched_track_idx, unmatched_det_idx).
    """
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    log_on = COST_LOG["on"] and (COST_LOG["frame"] or 0) <= COST_LOG["max_frame"]
    log_start = len(COST_LOG["rows"])
    log_ix = {}

    cost = np.full((len(tracks), len(dets)), 1e6)
    for i, (tr, pred) in enumerate(zip(tracks, preds)):
        # BoT-SORT: shift the prediction by the estimated camera motion before
        # comparing. Under a pan every prediction is wrong in the same direction,
        # which is exactly the error a global offset removes.
        px, py = pred[0] + cam[0], pred[1] + cam[1]
        gate = gate_width(tr.h, dt)
        for j, d in enumerate(dets):
            fx, fy = foot(d)
            dist = float(np.hypot(fx - px, fy - py))
            rec = None
            if log_on:
                rec = {"frame": COST_LOG["frame"], "stage": COST_LOG["stage"],
                       "track": tr.id, "det": j,
                       "dist": round(dist, 5), "gate": round(gate, 5),
                       "tr_kit": tr.kit, "d_kit": d["kit"].strip().lower(),
                       "tr_kit_votes": dict(tr.kit_votes),
                       "tr_num": tr.number, "d_num": d.get("num"),
                       "tr_h": round(float(tr.h), 4),
                       "d_h": round(float(d.get("h") or 0.0), 4),
                       "det_x": round(fx, 4), "det_y": round(fy, 4),
                       "pred_x": round(px, 4), "pred_y": round(py, 4),
                       "kit_mul": 1.0, "num_mul": 1.0, "h_mul": 1.0,
                       "cost": None, "in_gate": bool(dist <= gate),
                       "taken": False}
                log_ix[(i, j)] = len(COST_LOG["rows"])
                COST_LOG["rows"].append(rec)
            if dist > gate:
                continue
            # ACCELERATION GATE. `pred` is the constant-velocity prediction, so
            # (det - pred) IS the second difference of position: accepting this
            # detection would imply exactly this much change in velocity. Divide
            # by the track's apparent height for depth invariance and by dt²
            # for acceleration. See MAX_BODY_HEIGHT_ACCEL for the calibration.
            if ACCEL_GATE_ON[0] and tr.hits >= ACCEL_GATE_MIN_HITS:
                # Residual from the constant-velocity prediction, in body
                # heights. NO dt term: `pred` already extrapolated over dt, so
                # dividing again is what let a dropped frame hide a fly-out.
                r_bh = float(np.hypot((fx - px) * FRAME_ASPECT, fy - py)) \
                    / max(tr.h, 0.02)
                if rec:
                    rec["resid_bh"] = round(r_bh, 3)
                if r_bh > MAX_RESIDUAL_BH:
                    ACCEL_REJECTS.append(
                        {"frame": COST_LOG.get("frame"), "track": tr.id,
                         "resid": round(r_bh, 2), "dist": round(dist, 4),
                         "h": round(float(tr.h), 4)})
                    continue
            c = dist
            # Kit disagreement is a strong hint, not a law. As a hard veto it
            # killed roughly one track per frame: a player seen as "white" in one
            # frame and "blue" in the next — motion blur, shadow, a turned back —
            # became unmatchable and the track died even though the geometry was
            # unambiguous. A heavy multiplier keeps the preference (a genuine
            # opponent will lose to the right player every time) while letting a
            # single flickered colour word be outvoted by position.
            ta, tb = team_of(tr.kit), team_of(d["kit"].strip().lower())
            if ta is not None and tb is not None and ta != tb:
                c *= KIT_MISMATCH_PENALTY
                if rec:
                    rec["kit_mul"] = KIT_MISMATCH_PENALTY
            if tr.number is not None and d["num"] is not None \
                    and int(d["num"]) == tr.number:
                c *= NUMBER_MATCH_BONUS
                if rec:
                    rec["num_mul"] = NUMBER_MATCH_BONUS
            # BOX HEIGHT AS A DEPTH CUE. Apparent height has been smoothed on
            # every track since D12 and never used to decide anything.
            #
            # This is the one signal that survives a crossing. Two players at the
            # same screen position are at different DEPTHS, and depth shows up as
            # apparent height. Measured on the basketball swap at t+0.6s: the
            # candidates around the two tracks ranged 0.141 to 0.240 in height,
            # a 1.7x spread, while the tracks themselves sat at 0.176 and 0.207.
            # Position could not separate them — separation fell to 0.034 — and
            # height could.
            #
            # A ratio, not a difference, because apparent height scales with
            # distance: 0.02 apart means nothing near the camera and everything
            # far from it. Mild, because a real player's box height is noisy
            # frame to frame, and this must not out-shout position the way the
            # kit veto once did.
            if tr.h and d.get("h"):
                ratio = max(tr.h, d["h"]) / max(min(tr.h, d["h"]), 1e-4)
                hm = 1.0 + HEIGHT_MISMATCH_W * (ratio - 1.0)
                c *= hm
                if rec:
                    rec["h_mul"] = round(hm, 4)
            if rec:
                rec["cost"] = round(c, 5)
            cost[i, j] = c

    # PAD THE MATRIX so "no match" is an option the solver can actually choose.
    #
    # linear_sum_assignment on a SQUARE matrix must return a PERFECT matching.
    # With 10 tracks and 10 detections every track takes a detection whether one
    # fits or not — "this track coasts" and "this detection is a new player" were
    # not in the solution space at all. Out-of-gate cells hold 1e6, which is
    # FINITE, so a 4x kit mismatch at cost 0.41 is a bargain beside it.
    #
    # That is what caused the basketball identity swap, and it is not a kit
    # penalty problem. At frame 108 track 9 had a same-kit detection at cost
    # 0.0373 in gate and took an opponent's at 0.4061 — 2.3x farther, 10.9x
    # dearer — because that detection was in gate for NO OTHER track, so the
    # solver's only alternative for its column was 1e6. It then redistributed
    # the rest and pushed a white track onto a blue detection as well: a mutual
    # exchange, two kit-mismatched pairs accepted in one frame. Across frames
    # 0-300, 43 of 483 accepted pairs (8.9%) crossed a team boundary.
    #
    # Raising KIT_MISMATCH_PENALTY cannot fix this — the competing price was
    # 1e6, so it would need ~1e7, at which point it IS a hard veto and D12's
    # one-track-death-per-frame comes back. The post-filter below cannot either:
    # it only rejects a pair whose OWN cell is the sentinel, never one forced by
    # a sentinel elsewhere in the matrix.
    #
    # Pricing "unmatched" in gate widths keeps it commensurate with distance and
    # scales with apparent size, so a near player and a far one are judged alike.
    n, m = len(tracks), len(dets)
    big = np.full((n + m, m + n), 1e6)
    big[:n, :m] = cost
    big[np.arange(n), m + np.arange(n)] = [
        NO_MATCH_GATES * gate_width(tr.h, dt) for tr in tracks]        # coast
    big[n + np.arange(m), np.arange(m)] = [
        NO_MATCH_GATES * gate_width(float(d.get("h") or 0.02), dt)
        for d in dets]                                                 # birth
    big[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(big)
    pairs, mt, md = [], set(range(n)), set(range(m))
    for r, c in zip(rows, cols):
        if r < n and c < m and cost[r, c] < 1e5:
            pairs.append((r, c))
            mt.discard(r)
            md.discard(c)
    if log_on:
        for r, c in pairs:
            k = log_ix.get((r, c))
            if k is not None and k >= log_start:
                COST_LOG["rows"][k]["taken"] = True
    return pairs, sorted(mt), sorted(md)


def run(data, debug=False):
    frames = sorted(data["frames"], key=lambda f: f["frame"])
    src_fps = data.get("source_fps", 30)
    if not frames:
        sys.exit("no successful frames in that detections file")

    # ---- global kit -> team map (D6) -------------------------------------
    # Each call named colours independently; nothing said which team is which.
    # The two most-seen colours are the two teams. Anything else — goalkeeper,
    # a referee that slipped through, an unstable colour word — becomes None and
    # is tracked but never gated on and never given a team colour.
    kit_counts = Counter(p["kit"].strip().lower()
                         for f in frames for p in f["players"])
    teams = [k for k, _ in kit_counts.most_common(2)]
    team_map = {k: ("A" if i == 0 else "B") for i, k in enumerate(teams)}

    def team_of(kit):
        return team_map.get(kit)

    # Accent colour per kit, voted across frames. The renderer needs a second
    # colour to fall back on when the two kits are too close to tell apart.
    # Tolerates the older schema where `kits` was a plain list of colour words.
    accent_votes = defaultdict(Counter)
    for f in frames:
        for k in (f.get("kits") or []):
            if isinstance(k, dict) and k.get("colour") and k.get("accent"):
                accent_votes[k["colour"].strip().lower()][
                    str(k["accent"]).strip().lower()] += 1
    accents = {c: v.most_common(1)[0][0] for c, v in accent_votes.items() if v}

    # How far apart players stand in this clip: the 25th percentile of each
    # detection's distance to its nearest neighbour, over every frame.
    _sp = []
    for f in frames:
        pts = [foot(q) for q in f["players"]]
        for i, (x, y) in enumerate(pts):
            o = [np.hypot(u - x, v - y) for j, (u, v) in enumerate(pts) if j != i]
            if o:
                _sp.append(min(o))
    CLIP_SPACING[0] = sorted(_sp)[len(_sp) // 4] if len(_sp) >= 40 else None

    active, finished, cuts, degenerate = [], [], [], []
    cam = (0.0, 0.0)
    cam_cum, cam_at = [0.0, 0.0], {}
    prev_t, prev_scene = None, None
    prev_med_h, prev_n = None, None
    prev_kits = Counter()
    # Association health, recorded every frame. Added after a run produced 392
    # tracks for 22 players and three separate theories about why, none of them
    # measured. The distribution of these two numbers says which stage is
    # failing without any theorising at all.
    diag = {"rate": [], "births": [], "kit_blocked": [], "gate_missed": []}
    # The interval between the frames we actually sampled — 0.1s at 10fps from a
    # 30fps source. Several thresholds are "a few sampling intervals" and using
    # source frames for them is a unit error.
    nominal_dt = max(1.0 / max(data.get("fps") or src_fps, 1), 1e-3)

    for f in frames:
        fi = f["frame"]
        t = fi / src_fps
        dt = (t - prev_t) if prev_t is not None else (1.0 / src_fps)
        dt = max(dt, 1e-3)

        preds = [tr.kf.predict(dt) for tr in active]
        raw_dets = f["players"]
        dets = dedupe(raw_dets)

        # A frame the model got stuck on carries no information — skip it.
        #
        # Frame 81 returned FIFTEEN copies of one player: identical x, y, w, h
        # and confidence to three decimals. That is an autoregressive repetition
        # loop, and it is a real failure mode of constrained generation — the
        # schema demands an array, the model has nothing more to say, and it
        # fills the array by repeating itself. Well-formed, schema-valid,
        # meaningless.
        #
        # Deduplication correctly reduced it to one detection, at which point
        # one detection against fifteen live tracks looked exactly like a scene
        # change and the cut detector retired every track in the clip. Three
        # output frames rendered with a single marker.
        #
        # Treat it as a dropped frame instead: the tracker coasts across, which
        # is what it already does for the eight deadline drops in this run.
        if len(raw_dets) >= 4 and len(dets) / len(raw_dets) < DEGENERATE_RATIO:
            degenerate.append({"frame": fi, "raw": len(raw_dets),
                               "unique": len(dets)})
            prev_t = t
            continue

        # ---- D8: camera cuts, checked BEFORE association -------------------
        #
        # The previous detector used association collapse — "fewer than 30% of
        # active tracks found a match" — and was removed on 28 Aug after 52 false
        # positives across nine runs, every one landing on a corrupted-coordinate
        # frame. Disabling it took identities 49 -> 32. The scene-sentence
        # discontinuity it was supposed to fall back on is no better: measured on
        # the clip with three verified cuts, word overlap between consecutive
        # scene sentences is 0.26 at a cut against 0.36 away from one, and the
        # no-cut control spans the same 0.09-0.44 range throughout. The model
        # rewrites its sentence every frame regardless of what the camera did.
        #
        # SHOT SCALE is the signal. A cut moves the camera, so apparent player
        # size changes violently; a pan does not. Measured on football_cuts
        # against allstars (30s, no cuts) as the control:
        #
        #   t+17.2s cut   median box height 0.108 -> 0.825   ratio 6.64   dn 8
        #   t+21.6s cut                     0.745 -> 0.076   ratio 0.90   dn 12
        #   t+29.4s cut                     0.124 -> 0.976   ratio 6.54   dn 7
        #   allstars, entire clip, no cuts             max ratio 0.19   max dn 3
        #
        # Three of four cuts caught, zero false positives on the control. The
        # fourth (t+18.4s, keeper close-up -> behind-goal angle) is invisible to
        # this and to everything else we have: both shots are tight, so the scale
        # barely moves. A cut between two similarly-scaled shots is not
        # detectable from detections alone, and that is a stated limitation.
        #
        # Deliberately NOT using association collapse: it is the signal that
        # produced the 52 false positives, and a degenerate frame looks exactly
        # like it. Shot scale is computed from the detections only and cannot be
        # confused with a frame the model got stuck on.
        med_h = float(np.median([p["h"] for p in dets])) if dets else 0.0
        # max(), not len(active): the t+21.6s cut goes FROM a 3-player goalmouth
        # close-up TO a 15-player wide shot. Guarding on live tracks alone missed
        # it, because the shot being left had almost nothing in it. A jump from
        # 3 tracks to 15 detections is the strongest cut evidence in the clip;
        # the guard exists to avoid calling a cut on thin evidence, and evidence
        # on either side of the boundary counts.
        if prev_med_h and med_h and prev_n:
            h_ratio = abs(med_h - prev_med_h) / max(prev_med_h, 1e-4)
            d_n = abs(len(dets) - (prev_n or 0))
            n_ratio = max(len(dets), prev_n) / max(min(len(dets), prev_n), 1)
            kb = Counter((p.get("kit") or "").strip().lower() for p in dets)
            allk = set(prev_kits) | set(kb)
            kit_l1 = sum(abs(prev_kits.get(k, 0) / max(prev_n, 1)
                             - kb.get(k, 0) / max(len(dets), 1)) for k in allk)
            # CUT_MIN_TRACKS still guards the two ABSOLUTE tests, which are the
            # ones that go wrong on thin evidence. The ratio tests are scale-free
            # and are exactly what the guard was wrongly suppressing at the
            # 1-player-to-4-player boundary, so they run unguarded.
            big_enough = max(len(active), len(dets)) >= CUT_MIN_TRACKS
            # AT LEAST TWO SIGNALS, added 5 Sep. Any one test on its own is a
            # false-positive generator: the v4 run fired a cut at frame 690 on
            # h_ratio 0.77 alone, with the SAME 15 players, an identical kit mix
            # (kit_L1 0.00) and the model's own scene sentence describing the
            # same wide shot on both sides. Box heights simply wobbled.
            #
            # A real cut changes more than one thing at once. Measured on every
            # true cut in the set:
            #   516  h 6.64  n 9.00  kit 1.78   -> 3 signals
            #   552  h 0.09  n 4.00  kit 1.50   -> 2
            #   648  h 0.90  n 5.00  kit 0.67   -> 2
            #   882  h 6.28  n 8.00  kit 1.50   -> 3
            #   690  h 0.77  n 1.00  kit 0.00   -> 1   <- the false positive
            #
            # Every genuine cut corroborates; the false one does not. Costs
            # nothing on the cut-free clips, where no boundary trips even one.
            sig = ((1 if (big_enough and (h_ratio > CUT_SCALE_JUMP
                                          or d_n >= CUT_COUNT_JUMP)) else 0)
                   + (1 if n_ratio >= CUT_COUNT_RATIO else 0)
                   + (1 if kit_l1 > CUT_KIT_L1 else 0))
            fires = sig >= 2
            recent = cuts and (fi - cuts[-1]["frame"]) < CUT_DEBOUNCE_S * src_fps
            if fires and recent:
                # Same event firing twice. Replace the earlier call if this
                # boundary has the stronger evidence, rather than keeping
                # whichever happened to come first.
                if n_ratio > cuts[-1].get("n_ratio", 0):
                    cuts.pop()
                    for tr in active:
                        if tr.hits >= MIN_HITS:
                            finished.append(tr)
                    active, preds = [], []
                else:
                    fires = False
            if fires:
                cuts.append({"frame": fi, "h_ratio": round(h_ratio, 2),
                             "d_n": d_n, "n_ratio": round(n_ratio, 2),
                             "kit_l1": round(kit_l1, 2),
                             "med_h": round(med_h, 4),
                             "prev_med_h": round(prev_med_h, 4)})
                # Retire every track rather than let it reach across the cut.
                # Identity is re-anchored afterwards by (kit, number): two
                # fragments that read the same number on the same kit are the
                # same player, which needs no special case here.
                for tr in active:
                    if tr.hits >= MIN_HITS:
                        finished.append(tr)
                active, preds = [], []
        if dets:
            prev_kits = Counter((q.get("kit") or "").strip().lower()
                                for q in dets)
        prev_med_h, prev_n = (med_h or prev_med_h), len(dets)

        # ---- camera motion, estimated BEFORE anything is associated -------
        # Chicken-and-egg resolved: vote on the shift straight from the point
        # sets, then associate with it already applied.
        max_shift = MAX_BODY_HEIGHTS_PER_SEC * 0.14 * dt + PAN_SEARCH_FLOOR
        cx, cy, support = estimate_global_motion(preds, dets, max_shift)
        # Measured: the real global shift on this footage is 0.0065 median and
        # 0.0508 at worst, comfortably inside the association gate. Compensation
        # is therefore nearly free to skip and actively harmful when the vote is
        # weak, because a noisy offset drags every prediction sideways. Apply it
        # only when a clear majority of tracks agree on the same shift.
        cam = (cx, cy) if support >= 0.50 else (0.0, 0.0)
        # Accumulate it. A painted marking sits still ON THE PITCH, not on the
        # screen — under a pan it slides across the frame like everything else.
        # Summing the per-frame camera translation gives a roughly
        # camera-stabilised coordinate system, which is the only frame in which
        # "this position keeps coming back" means anything.
        cam_cum[0] += cam[0]
        cam_cum[1] += cam[1]
        cam_at[fi] = (cam_cum[0], cam_cum[1])

        # ---- D8 cut detection: REMOVED 31 Aug ----------------------------
        #
        # It never once fired on a real shot change, because neither recorded
        # clip contains one. Every cut it did fire — 52 across nine runs — was a
        # false positive, and on the good-quality runs each one landed exactly
        # on a frame whose coordinates were corrupted, plus the recovery frame
        # after it. A shot change is a property of the footage; it cannot appear
        # at 1080p and not at 720p on the same ten seconds.
        #
        # Measured cost on the shipped model, gemini-3.7-flash 30s/1080:
        #     CUT_MATCH_RATE = 0.30  ->  49 identities
        #     CUT_MATCH_RATE = 0.0   ->  32 identities   (detector disabled)
        # Two false cuts were costing 17 identities, a third of the total.
        #
        # The mechanism it was built for is real and will be needed for a clip
        # that actually contains a cut — retire every track, re-anchor by jersey
        # number — but it cannot be validated against footage with no cuts in
        # it. Rebuild it when we have such a clip, and test it on that clip.

        # ---- ByteTrack two-stage association ------------------------------
        hi = [d for d in dets if d["conf"] >= HIGH_CONF]
        lo = [d for d in dets if d["conf"] < HIGH_CONF]

        COST_LOG["frame"], COST_LOG["stage"] = fi, "hi"
        pairs, un_tr, un_hi = match(active, preds, hi, dt, cam, team_of)
        residuals = []
        for ti, di in pairs:
            tr, d = active[ti], hi[di]
            fx, fy = foot(d)
            residuals.append((fx - preds[ti][0], fy - preds[ti][1]))
            gap = t - tr.last_t
            # OC-SORT's retro-correction is for a track returning from a LONG
            # absence. The threshold was written as `2.5 / src_fps` — 2.5 SOURCE
            # frames, 0.083s — while the normal gap between consecutive samples
            # at 10fps is 0.1s. So it fired on every ordinary update, replacing
            # the Kalman velocity with a raw two-point difference every single
            # frame and feeding that noise straight into the next prediction.
            # The unit that matters is the SAMPLING interval, not the source one.
            if gap > 2.5 * nominal_dt:
                tr.kf.retro_correct((fx, fy), gap)
            tr.kf.update((fx, fy), d["conf"], d["h"], d.get("w"))
            tr.hits += 1
            tr.absorb(d, fi, t)

        # Second pass: tracks that found nobody confident may still claim a
        # hesitant detection. This is where occluded players get kept alive.
        if un_tr and lo:
            sub = [active[i] for i in un_tr]
            subpred = [preds[i] for i in un_tr]
            COST_LOG["stage"] = "lo"
            p2, un2, _ = match(sub, subpred, lo, dt, cam, team_of)
            claimed = set()
            for si, di in p2:
                tr, d = sub[si], lo[di]
                fx, fy = foot(d)
                tr.kf.update((fx, fy), d["conf"], d["h"], d.get("w"))
                tr.hits += 1
                tr.absorb(d, fi, t)
                claimed.add(un_tr[si])
            un_tr = [i for i in un_tr if i not in claimed]

        # ---- camera motion, from detections only (BoT-SORT, no pixels) ----
        if len(residuals) >= 3:
            arr = np.array(residuals)
            cam = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
        else:
            cam = (cam[0] * 0.5, cam[1] * 0.5)   # decay when we cannot measure

        # ---- association health -------------------------------------------
        if active:
            diag["rate"].append(len(pairs) / len(active))
            diag["births"].append(len(un_hi))
            # Why did the leftovers not match? Separate a geometry failure from
            # a kit-gate veto: re-score the same pairs with the kit gate off and
            # see how many suddenly land inside the gate.
            kb = gm = 0
            for i in un_tr:
                tr, (px, py) = active[i], preds[i]
                px, py = px + cam[0], py + cam[1]
                gate = gate_width(tr.h, dt)
                near = [d for d in hi
                        if np.hypot(foot(d)[0] - px, foot(d)[1] - py) <= gate]
                if not near:
                    gm += 1
                elif all(team_of(tr.kit) is not None
                         and team_of(d["kit"].strip().lower()) is not None
                         and team_of(tr.kit) != team_of(d["kit"].strip().lower())
                         for d in near):
                    kb += 1
            diag["kit_blocked"].append(kb)
            diag["gate_missed"].append(gm)

        for i in un_tr:
            active[i].kf.coast()

        # ---- birth and death ----------------------------------------------
        for di in un_hi:
            nt = Track(hi[di], fi, t)
            active.append(nt)
            if COST_LOG["on"] and fi <= COST_LOG["max_frame"]:
                fx, fy = foot(hi[di])
                COST_LOG["rows"].append(
                    {"frame": fi, "stage": "birth", "track": nt.id, "det": di,
                     "d_kit": hi[di]["kit"].strip().lower(),
                     "d_num": hi[di].get("num"),
                     "det_x": round(fx, 4), "det_y": round(fy, 4),
                     "d_h": round(float(hi[di].get("h") or 0.0), 4)})
        still = []
        for tr in active:
            # Two ways to die now: age, and leaving the picture. The second only
            # applies while UNMATCHED — a track with a live detection near the
            # edge is a player standing on the touchline, which is not the same
            # as a coasted prediction that has sailed past it.
            cx, cy = float(tr.kf.x[0]), float(tr.kf.x[1])
            gone = (t > tr.last_t) and not (-EDGE_MARGIN <= cx <= 1 + EDGE_MARGIN
                                            and -EDGE_MARGIN <= cy <= 1 + EDGE_MARGIN)
            if gone:
                if tr.hits >= MIN_HITS:
                    finished.append(tr)
                continue
            if t - tr.last_t <= MAX_COAST_S:
                still.append(tr)
            elif tr.hits >= MIN_HITS:
                finished.append(tr)
        active = still

        prev_t, prev_scene = t, f.get("scene", "")

    finished.extend(tr for tr in active if tr.hits >= MIN_HITS)

    # ---- D7 + D8: label once, and re-anchor across cuts ------------------
    # Grouping by (kit, number) is what stitches a track that died at a cut back
    # to the track that replaced it. No special case needed: two fragments that
    # read the same number on the same kit ARE the same player.
    # A merge is only legitimate for track fragments that are separated IN TIME.
    # Two tracks alive in the same frame are two different people, whatever
    # number was read off them — so a misread, or two players genuinely wearing
    # numbers the model confused, must not collapse into one identity.
    #
    # It did, and the symptom was spectacular: identity A23 appeared TWICE in
    # every frame between 1.8s and 2.7s, 0.177 fraction units apart, so the two
    # markers read as one ring teleporting back and forth eleven times a second.
    # That is the flicker, and it was never a tracking failure at all — the
    # tracks were fine, the labelling pass glued them together.
    def overlaps(a, b):
        return not (a.obs[-1][0] < b.obs[0][0] or b.obs[-1][0] < a.obs[0][0])

    groups, singles = {}, []
    for tr in finished:
        n = tr.number
        if n is None:
            singles.append(tr)
            continue
        bucket = groups.setdefault((tr.kit, n), [])
        for members in bucket:
            if not any(overlaps(tr, other) for other in members):
                members.append(tr)
                break
        else:
            bucket.append([tr])          # a new, concurrent wearer of that number

    identity, display = {}, {}
    for (kit, num), buckets in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        for k, members in enumerate(buckets):
            # Concurrent wearers of the same number get suffixed ids rather than
            # being merged. Rare, and visible when it happens, which is correct:
            # it means the number was misread and the viewer should see two.
            did = f"{team_map.get(kit, '?')}{num}" + ("" if k == 0 else f"^{k}")
            for tr in members:
                identity[tr.id] = did
                display[did] = {"team": team_map.get(kit), "kit": kit,
                                "number": num,
                                "number_source": "read" if k == 0 else "contested"}
    # Invented labels are LETTERS, not numbers. Drawing them as pale numbers was
    # unreadable as a distinction — at 12px a desaturated 57 is just a 57 you
    # squint at, and a viewer has no reason to guess it means "we made this up".
    # Jerseys carry digits, so a letter is unambiguous on sight and needs no
    # legend, no opacity trick and no colour change.
    for k, tr in enumerate(sorted(singles, key=lambda x: x.obs[0][0])):
        did = f"{team_map.get(tr.kit, '?')}·{tr.id}"
        identity[tr.id] = did
        display[did] = {"team": team_map.get(tr.kit), "kit": tr.kit,
                        "number": letter_label(k), "number_source": "fallback"}

    # ---- interpolate every track up to the full source frame rate --------
    # Written by detect.py from ffprobe. Falling back to the last sampled index
    # would silently truncate the tail of the video — at 10fps the last sample
    # is frame 897, not 899, and three frames would go unrendered.
    total = data.get("n_source_frames") or (frames[-1]["frame"] + 1)
    sample_step = max(1, round(src_fps / max(data.get("fps") or src_fps, 1)))
    per_frame = defaultdict(list)
    short_tracks = 0
    cut_frames_draw = sorted(c["frame"] for c in cuts)
    for tr in finished:
        if len(tr.obs) < 2:
            continue
        # MINIMUM DRAW LIFETIME. Being real enough to track (MIN_HITS) and being
        # worth showing are different bars, and they were the same number. A
        # track seen only three times is where the spurious births live - A-4 on
        # allstars appears at t+0.00s, its x reverses -0.34 then +0.35 body
        # heights, and it is gone by t+0.57s having never corresponded to a
        # player. The round-trip test above cannot help: with three points there
        # is no clean pair to judge the middle one against.
        # NB: tested AFTER reject_outliers below, not here, because the bar
        # belongs on observations that SURVIVE the outlier test. Track 22 on
        # allstars had 4 raw observations - 0.009, 0.013, 0.074, 0.134 - of
        # which the last was rejected as unreachable, leaving 3 real ones and a
        # marker drifting off a player it had already lost. Counting the raw
        # four let it draw; counting the surviving three does not.
        # REMOVE the bad observation; do not cut the track around it.
        #
        # Splitting was the wrong instrument. A break becomes a hole in the
        # rendered path, any fragment under two observations is dropped
        # entirely, and the result was markers blinking out several times a
        # second — which watches worse than the glide it was meant to prevent.
        # Judged side by side, a rare smooth glide beats constant blinking.
        #
        # The right test is the one already used on the ball: an observation
        # that is implausible to reach AND implausible to leave, but whose
        # neighbours are perfectly plausible without it, is a bad association,
        # not motion. Drop the point and interpolate straight through the hole,
        # so the path stays continuous and the excursion never happens.
        o = reject_outliers(sorted(tr.obs), src_fps)
        if len(o) < MIN_DRAW_SAMPLES:
            short_tracks += 1
            continue
        if len(o) < 2:
            continue
        # Outliers first, then smooth. The other order would let a bad
        # association drag its neighbours toward it before being removed.
        if RECONFIRM[0]:
            o = drop_unconfirmed(o, nominal_dt * src_fps)
            if len(o) < 2:
                continue
        o = smooth_observations(o, src_fps, nominal_dt)
        # NO gap cap. Coasting across an unobserved stretch was capped at 0.35s
        # and then 0.2s to stop markers being drawn for players nobody had
        # detected. Judged on screen, both were worse than not capping: a marker
        # that holds through a brief absence hides the detector's misses, while
        # one that stops draws attention to them. The interpolated marker is
        # usually in about the right place — the player did not teleport — so
        # the "phantom" is closer to the truth than the hole it leaves.
        #
        # This is a deliberate choice of a *plausible* artifact over an
        # *honest* one, for a deliverable that is watched rather than audited.
        # Worth stating plainly in the report rather than hiding.
        # SPLIT ON A LONG GAP. This was `segs = [o]` — one segment for the whole
        # observation list — so np.interp drew a straight line across every gap
        # however long, and however far apart its endpoints were. A track that
        # lost a player at the frame edge and later re-acquired one somewhere
        # else glided its marker across the pitch through empty space. Measured
        # markers travelling 0.09-0.17 fraction units in their final 0.6s, which
        # is the "ring flying out" the user reported. It was never coasting or
        # Kalman extrapolation; it was interpolation joining two distant dots.
        #
        # The note above still stands for SHORT gaps: a marker that holds through
        # a brief absence is closer to the truth than the hole it leaves, because
        # the player did not teleport. Across a long gap that reasoning inverts —
        # the player may well have gone, and a marker sliding smoothly to wherever
        # the track resumed is a confident lie rather than a plausible guess.
        # Split on DISTANCE as well as time. MAX_DRAW_GAP_S was only ever a proxy
        # for "these two dots are too far apart to join", and a poor one: at 5fps
        # a single missed sample is 0.4s, comfortably inside the 0.50s limit, so
        # every visible glide in every clip is exactly 12 source frames long.
        # Worst observed 0.143 fraction units — 183px at 1280 wide — travelled in
        # 0.4s, which no footballer does.
        #
        # MAX_PLAYER_SPEED already encodes what a player can manage. Using it
        # here refuses the join when the endpoints are further apart than the gap
        # allows, while keeping every benign 0.03-0.05 bridge that stops a marker
        # blinking through a one-frame miss. Measured cost: 24 to 60 player-frames
        # per clip, under 0.25% of draws on the football clips, against 264-811
        # for the blunt alternative of tightening MAX_DRAW_GAP_S to 0.30.
        segs, cur = [], [o[0]]
        for a, b in zip(o, o[1:]):
            span = b[0] - a[0]
            gap_s = span / src_fps
            # The distance test applies ONLY where a sample was actually missed.
            # Between adjacent samples there is nothing to interpolate across, so
            # the only thing it can do there is fragment a track on ordinary fast
            # motion — a player crossing 0.05 units in 0.2s under a panning
            # camera is unremarkable, and testing every pair cut 384 markers on
            # allstars against the 36 that the bridged gaps actually account for.
            bridged = span > sample_step
            moved = float(np.hypot(b[1] - a[1], b[2] - a[2]))
            if gap_s > MAX_DRAW_GAP_S or (bridged and moved > MAX_PLAYER_SPEED * gap_s):
                segs.append(cur)
                cur = [b]
            else:
                cur.append(b)
        segs.append(cur)
        did = identity[tr.id]
        info = display[did]
        # EVERY coherent stretch is drawn, not just the longest. A break means
        # "do not draw a line across this gap", not "forget the rest of this
        # player" — keeping only the longest segment halved the markers on
        # screen, from 18 per frame to 8.
        for si, seg in enumerate(segs):
            if len(seg) < 2:
                continue
            idx = np.array([p[0] for p in seg], dtype=float)
            lo_f, hi_f = int(idx[0]), int(idx[-1])
            # Hold the last observation forward one sampling interval, so the
            # closing frames of the clip are not empty (the last sample at 10fps
            # from 30fps lands on frame 897, not 899).
            hi_f = min(hi_f + sample_step - 1, total - 1)
            grid = np.arange(lo_f, hi_f + 1)
            xs = np.interp(grid, idx, [p[1] for p in seg])
            ys = np.interp(grid, idx, [p[2] for p in seg])
            # EASING (item 17). Linear interpolation is C0: velocity is constant
            # inside each 0.2s span and changes INSTANTANEOUSLY at every sample,
            # so the drawn path is a polyline that turns a corner six frames
            # apart. A critically damped follower is used rather than a spline
            # because it cannot overshoot by construction - and this project has
            # already been bitten by an overshooting centred smoother, which
            # pushed every marker AHEAD of its player under an accelerating pan.
            # The cost is a small constant lag while a player accelerates.
            if SMOOTH_FOLLOW_S > 0:
                xs, ys = damped_follow(xs, ys, 1.0 / src_fps, SMOOTH_FOLLOW_S)
            ws = np.interp(grid, idx, [p[3] for p in seg])
            hs = np.interp(grid, idx, [p[4] for p in seg])
            for k, fr in enumerate(grid):
                # `team`, `label` and `read` are denormalised onto every
                # player-frame on purpose. The renderer should never have to
                # parse an id string or cross-reference a table to know what to
                # draw — that coupling is how it once rendered every marker grey.
                # FADE (item 16). The last FADE_OUT_S of every drawn stretch
                # ramps to zero, so a marker is already invisible by the frame
                # it would otherwise pop out of existence. Applied to the END OF
                # THE DRAWN SPAN rather than triggered on track death, which is
                # only possible because this is a batch pipeline: we already
                # know which tracks never come back. A player who walks off the
                # edge and one lost behind a crowd therefore leave identically.
                # NOT ACROSS A CUT. A fade is a statement that the player is
                # leaving; at a cut the whole SCENE leaves, and a marker easing
                # out over the next 9 frames is drawn on top of a different
                # camera angle. Measured on football_cuts: 20 of 62 drawn
                # stretches end within 10 frames of a cut, so a third of all
                # fades were painting stale markers onto new footage. Those end
                # hard, at full opacity, which is what a cut looks like.
                _fade_n = FADE_OUT_S * src_fps
                _left = len(grid) - 1 - k
                _at_cut = any(0 <= c - int(grid[-1]) <= sample_step + 1
                              for c in cut_frames_draw)
                _fade = (1.0 if (_fade_n <= 0 or _at_cut)
                         else min(1.0, _left / _fade_n))
                per_frame[int(fr)].append({
                    "id": did, "track": tr.id, "fade": round(_fade, 3),
                    "team": info["team"], "label": str(info["number"]),
                    "read": info["number_source"] == "read",
                    "x": round(float(xs[k]), 5), "y": round(float(ys[k]), 5),
                    "w": round(float(ws[k]), 5), "h": round(float(hs[k]), 5)})

    # ---- the ball --------------------------------------------------------
    # Recall is not the problem — 97% of frames returned a ball. Precision is.
    # A football is a small white round thing on a pitch covered in other small
    # white round things: the penalty spot, the centre spot, a boot, a patch of
    # sock. The model reports these with high confidence because they genuinely
    # look like what was asked for. Two filters, both purely geometric.
    raw = [(f["frame"], f["ball"]["x"] + f["ball"]["w"] / 2,
            f["ball"]["y"] + f["ball"]["h"] / 2, f["ball"].get("conf", 1.0),
            f["ball"]["h"], f["ball"]["w"])
           for f in frames if f.get("ball")]

    ball_outliers = []

    # TRIED AND REMOVED 3 Sep: a wrong-sport filter. detect.py asked the model to
    # name which sport's ball each detection was, the clip's sport was the
    # majority verdict, and any minority report was rejected as a
    # misidentification. On basketball all 136 detections said "basketball",
    # including every one the geometric filters threw out, so it never fired.
    # The model names the sport it is watching rather than classifying the
    # object, which puts the field downstream of the error it was meant to catch.

    # Filter -1 — CONFIDENCE FLOOR, off unless asked for. The kinematic filters
    # below can only catch a decoy that moves implausibly. A white spot mark or
    # a boot sitting NEXT to the real ball implies a perfectly ordinary
    # 0.1-0.4 frac/s and sails through every one of them; on the cuts clip the
    # three decoys reported at t+14.4s, t+14.6s and t+25.0s had implied speeds of
    # 0.100, 0.305 and 0.771 against a gate of 1.44-1.56. Confidence is the only
    # signal that separates them: they came back at 0.70-0.80 where real balls
    # sit at 0.90-0.95.
    if BALL_MIN_CONF > 0:
        low = [r for r in raw if r[3] < BALL_MIN_CONF]
        for r in low:
            ball_outliers.append({"frame": r[0], "kind": "low-conf",
                                  "conf": r[3]})
        raw = [r for r in raw if r[3] >= BALL_MIN_CONF]

    # Filter 0 — THE FLAT BOX. This is the one that works, and it is geometry
    # rather than tracking.
    #
    # Every decoy on this clip is the same physical thing: a short white dash
    # painted on the turf. A mark lying flat on the grass is foreshortened
    # vertically by an oblique camera and not at all horizontally. A ball is a
    # sphere and keeps its vertical extent. So the box WIDTH of the two
    # populations is identical and the HEIGHT is not:
    #
    #     real ball      w med 0.0080   h med 0.0120
    #     painted dash   w med 0.0075   h med 0.0065
    #
    # Replicated on an independent API run of the same clip (0.0110 vs 0.0060),
    # and the detections it flags outside the windows reported by eye sit a
    # median 0.008 from a known decoy position against 0.193 for the detections
    # it keeps — they are the same painted marks, found without being told.
    #
    # Normalised by the clip's own median rather than fixed in fraction units,
    # so it survives a change of shot scale.
    if len(raw) >= 8:
        med_h = float(np.median([r[4] for r in raw]))
        flat = {r[0] for r in raw if r[4] < BALL_FLAT_H_FRAC * med_h}
        for fr in sorted(flat):
            ball_outliers.append({"frame": fr, "kind": "flat-box"})
        raw = [r for r in raw if r[0] not in flat]

    # RETIRED 28 Aug: the static-cluster filter that used to sit here, which
    # rejected positions reported repeatedly but sparsely in time.
    #
    # Attributed against a hand-labelled set it caught 10 painted marks and
    # destroyed 18 real detections doing it — worse than one for one. Among the
    # casualties were nine consecutive samples of a ball genuinely sitting still
    # at a player's feet, which is precisely the case its "sparse in time" test
    # was supposed to protect. Three attempts to rescue it with camera-motion
    # compensation (cumulative, differential, windowed) each made things worse.
    #
    # The flat-box test above identifies the same marks by what they ARE rather
    # than by where they keep turning up, so a positional heuristic has nothing
    # left to do here.

    # Filter 1 — the round trip. If the ball leaps away and is back next sample
    # where it started, the middle reading was a decoy, not motion. A real ball
    # travelling that fast keeps going; it does not return to its own launch
    # point one tenth of a second later. This is the boot-mistaken-for-ball case
    # and it shows up as a one-frame flicker in the video.
    # RUN TO A FIXED POINT, not once. The round trip only ever adapted on its
    # LEFT: `a = keep[-1]` is the last surviving detection, so a rejected
    # predecessor is skipped, but `c` comes from the unfiltered list and the
    # speed gate below runs afterwards, so nothing it removes ever feeds back.
    #
    # That order-dependence is what put a decoy on screen at allstars t+23.1s.
    # The model reported the same white spot on frames 678, 690, 696 and 702.
    # On the earlier run frame 690 came back with h=0.0090, two ten-thousandths
    # under the flat-box line, and was removed FIRST — so the round trip judged
    # 696 against 684 and 702, saw a clean leap-and-return, and rejected it. On
    # the later run the same detection came back at h=0.0110, survived flat-box,
    # and left 696 sitting 0.006 from its neighbour, which is not a leap at all.
    # The speed gate then removed 690 anyway, but too late: 696 was already kept,
    # and interpolation dutifully slid the ball down to the white spot and back.
    #
    # A second pass sees 684 and 714 as 696's neighbours and rejects it, which is
    # what the first run got by luck. Bounded, because this codebase has been
    # burned once by an over-eager ball filter: the static-cluster test retired on
    # 28 Aug caught 10 painted marks and destroyed 18 real detections doing it.
    # Passes stop as soon as a pass removes nothing, and never exceed the cap.
    # Local image scale per SOURCE FRAME, from the players the model reported
    # there. This is what turns both ball thresholds from flat frame fractions
    # into depth-aware ones; see BALL_JUMP_PH.
    frame_scale = {}
    for f in data.get("frames") or []:
        hs = sorted(p["h"] for p in (f.get("players") or []) if p.get("h"))
        if len(hs) >= BALL_SCALE_MIN_PLAYERS:
            frame_scale[f["frame"]] = hs[len(hs) // 2]

    def _scale(fi):
        """Median player height near this frame, or None if unusable."""
        if fi in frame_scale:
            return frame_scale[fi]
        near = [k for k in frame_scale if abs(k - fi) <= src_fps // 2]
        if not near:
            return None
        return frame_scale[min(near, key=lambda k: abs(k - fi))]

    for _pass in range(BALL_FILTER_PASSES):
        n_before = len(raw)
        if len(raw) >= 3:
            keep = [raw[0]]
            step_f = src_fps // max(data.get("fps") or 10, 1)
            skip_to = 0
            for i in range(1, len(raw) - 1):
                if i < skip_to:
                    continue            # already discarded as part of an excursion
                a, b, c = keep[-1], raw[i], raw[i + 1]
                if (b[0] - a[0]) > 2 * step_f + 1:
                    keep.append(b)          # too far apart in time to judge
                    continue
                d_ab = np.hypot(b[1] - a[1], b[2] - a[2])
                d_bc = np.hypot(c[1] - b[1], c[2] - b[2])
                d_ac = np.hypot(c[1] - a[1], c[2] - a[2])
                sc = _scale(b[0])
                jump = (min(BALL_JUMP_PH * sc, BALL_JUMP_FRAC)
                        if sc else BALL_JUMP_FRAC)
                # TWO-SAMPLE EXCURSIONS - TRIED 5 SEP AND REVERTED.
                #
                # The three-point test only sees a decoy lasting ONE sample: if
                # b and c are both on the same wrong object, d_bc is tiny and
                # the return leg has not happened yet. So I widened it to ask
                # whether dropping BOTH b and c reconciles a -> d.
                #
                # It removed the RIGHT detections and kept the wrong ones. On
                # allstars t+22.6-23.4s the ball reads 0.496,0.373 / 0.665,0.834
                # / 0.466,0.376 / 0.460,0.377 / 0.663,0.833 - two decoys at
                # (0.66,0.83) BRACKETING two real points. Asking "do the
                # endpoints agree while the middle strays" then indicts the real
                # pair, because the endpoints that agree are both the decoy.
                # Same on basketball at 15.40-15.60.
                #
                # The test is symmetric and local: it can find an excursion but
                # not which side of it is the truth. When decoys locally
                # outnumber real detections it inverts. A chord-deviation
                # variant fails identically - the chord is drawn between the
                # decoys. Deciding WHICH cluster is the ball needs continuity
                # against an established trajectory, i.e. an actual ball filter,
                # not a wider local window.
                if d_ab > jump and d_bc > jump and d_ac < d_ab * 0.5:
                    ball_outliers.append({"frame": b[0], "kind": "round-trip",
                                          "jump": round(float(d_ab), 3),
                                          "pass": _pass + 1})
                    continue
                keep.append(b)
            keep.append(raw[-1])
            raw = keep

        # Speed gate, inside the loop so its removals feed the next round trip.
        gated = []
        for r in raw:
            if gated:
                gap = max((r[0] - gated[-1][0]) / src_fps, 1e-3)
                speed = np.hypot(r[1] - gated[-1][1], r[2] - gated[-1][2]) / gap
                sc = _scale(r[0])
                base = (min(BALL_GATE_PH_PER_SEC * sc, BALL_GATE_PER_SEC)
                        if sc else BALL_GATE_PER_SEC)
                allowed = base * (0.5 + r[3])
                if speed > allowed:
                    ball_outliers.append({"frame": r[0], "kind": "over-speed",
                                          "frac_per_s": round(float(speed), 2),
                                          "conf": r[3], "pass": _pass + 1})
                    continue
            gated.append(r)
        raw = gated

        if len(raw) == n_before:
            break                       # converged

    # Filter 2 — the speed gate. Anything demanding a speed a struck ball cannot
    # reach is a different object, not a fast ball. Catches sustained drift onto
    # a static decoy, which the round-trip test cannot see because the decoy is
    # reported for several frames running.
    #
    # Confidence weights the gate rather than gating on its own. Measured on the
    # first clip: decoys sit at median conf 0.68 against 0.95 for real balls, so
    # there IS signal — but a hard threshold at 0.70 drops 20 of 37 decoys while
    # also losing 19 of 252 good detections. Roughly one-for-one, which is not
    # worth having. As a multiplier on the speed gate it costs nothing: a
    # confident detection earns more benefit of the doubt, a hesitant one has to
    # be geometrically plausible as well.
    # The gate itself now runs inside the fixed-point loop above, so that what it
    # removes is visible to the next round-trip pass. Nothing left to do here.
    seen = raw

    ball_at = {}
    if seen:
        bidx = np.array([s[0] for s in seen], dtype=float)
        bx = np.array([s[1] for s in seen])
        by = np.array([s[2] for s in seen])
        # Size travels with position now. `raw` has carried the ball's w and h
        # since the flat-box filter was written (r[4], r[5]) but they stopped
        # here, so the renderer drew a flat 7.7px dot for a ball the model
        # measured anywhere from 7.7px to 129px. Interpolated the same way as x
        # and y, so a bridged frame gets a plausible size rather than none.
        bh = np.array([s[4] for s in seen])
        bw = np.array([s[5] for s in seen])
        # A rejected frame IS still interpolated over, deliberately.
        #
        # Tried the opposite on 2 Sep — skip any frame a filter had vetoed, on the
        # reasoning that repainting a frame you judged bad undoes the filter's
        # decision. Measured across all four clips it was wrong, and backwards:
        # all 17 rejections were being bridged, and the interpolated point sat a
        # median 0.179 fraction units from the rejected detection (max 0.506)
        # against a player spacing of 0.072. The decoy COORDINATE was never being
        # drawn. Rejection removes the bad measurement; interpolation then supplies
        # a substitute from the surviving neighbours, which is the whole point.
        # Skipping them replaced 17 good positions with 17 holes and fixed nothing.
        # DO NOT INTERPOLATE ACROSS A CAMERA CUT. Player tracks have been
        # retired at a cut since D8 — every track is finished and identity is
        # re-anchored afterwards — but the ball path never read the cut list at
        # all, so np.interp happily drew the ball from wherever it was before the
        # cut to wherever it appeared after it. Across a POV change those two
        # points have no spatial relationship whatever, and the marker glides
        # between them through a scene that no longer exists.
        cut_frames = sorted(c["frame"] for c in cuts)

        def _crosses_cut(f0, f1):
            return any(f0 < c <= f1 for c in cut_frames)

        for a, b in zip(seen, seen[1:]):
            if (b[0] - a[0]) / src_fps > BALL_MAX_GAP_S:
                continue      # a long absence is a real absence — do not invent it
            if _crosses_cut(a[0], b[0]):
                continue
            for fr in range(a[0], b[0] + 1):
                ball_at[fr] = (float(np.interp(fr, bidx, bx)),
                               float(np.interp(fr, bidx, by)),
                               float(np.interp(fr, bidx, bw)),
                               float(np.interp(fr, bidx, bh)))
        for r in seen:
            ball_at[r[0]] = (r[1], r[2], r[5], r[4])
        # HOLD AN UNBRIDGED SIGHTING FOR THE SAMPLE INTERVAL IT REPRESENTS.
        #
        # A detection that cannot be joined to a neighbour - because the gap
        # exceeds BALL_MAX_GAP_S, or a cut sits between them - was drawn on ONE
        # source frame. At 5fps into 30fps that is 0.03s: a single-frame strobe
        # that reads as the marker glitching rather than as the ball being seen.
        # Measured on football_cuts v4 at t+25.60s and t+27.00s, both isolated
        # by 1.2-1.4s gaps and both drawn for exactly 1 frame.
        #
        # The sample stands for 0.2s of video, so hold it for 0.2s. This is the
        # same treatment the PLAYER path already gives its final observation,
        # and it invents nothing: the position is the one the model reported,
        # held for the interval it was sampled over, rather than extrapolated.
        # Never written over a bridged frame, and never across a cut.
        hold = src_fps // max(data.get("fps") or 10, 1)
        for r in seen:
            for k in range(1, hold):
                f2 = r[0] + k
                if f2 in ball_at or f2 >= total:
                    continue
                if _crosses_cut(r[0], f2):
                    break
                ball_at[f2] = (r[1], r[2], r[5], r[4])

    # ---- who is on the ball, smoothed ------------------------------------
    # Measured against the player's BOX, not their foot point.
    #
    # It used to be hypot(p.x - ball.x, p.y - ball.y), and p.x/p.y is the foot
    # point — the bottom-centre of the box, which is where the marker is drawn.
    # That is the right reference in football, where the ball is at the feet, and
    # systematically wrong in basketball, where it is held at chest or head
    # height. The handler's foot point sits 0.15-0.20 fraction units BELOW the
    # ball, so a defender standing slightly to the side can be nearer in straight
    # line distance and steal the marker. The user saw exactly that: the ring
    # following the man marking the ball carrier rather than the carrier.
    #
    # Two changes, both sport-agnostic, so no per-sport branch is needed:
    #
    #   1. CONTAINMENT WINS. If the ball's centre falls inside a player's box,
    #      that player has it. Measured earlier: 79% of basketball ball
    #      detections sit inside a box, and 23-40% in football. It was a useless
    #      rejection rule for precisely the reason it is a good possession rule.
    #   2. Otherwise, distance to the nearest point ON THE BOX rather than to the
    #      foot point. A ball at a footballer's feet is just below the box edge,
    #      so this barely moves football; it moves basketball a great deal.
    def _box(p):
        return (p["x"] - p["w"] / 2, p["y"] - p["h"], p["x"] + p["w"] / 2, p["y"])

    def _airborne(players, bx, by, b_h):
        """Is the ball above the pitch rather than on it?

        Fits foot_y = a*box_h + b over the players in this frame — that line is
        the ground plane in image space. The ball's own size gives its depth, so
        b_h / BALL_PLAYER_H gives the height a player at the ball's distance
        would have, and the fit turns that into where their feet would be.
        Returns False whenever the fit cannot be trusted, so a bad frame falls
        back to the previous 2D behaviour rather than suppressing possession.
        """
        if b_h <= 0 or len(players) < GROUND_FIT_MIN_PLAYERS:
            return False
        hs = np.array([p["h"] for p in players], dtype=float)
        fy = np.array([p["y"] for p in players], dtype=float)   # foot point
        if hs.max() - hs.min() < 1e-3:
            return False                       # no perspective spread to fit
        A = np.vstack([hs, np.ones(len(hs))]).T
        coef, *_ = np.linalg.lstsq(A, fy, rcond=None)
        pred = A @ coef
        ss_tot = float(((fy - fy.mean()) ** 2).sum())
        if ss_tot <= 0:
            return False
        r2 = 1.0 - float(((fy - pred) ** 2).sum()) / ss_tot
        if r2 < GROUND_FIT_MIN_R2:
            return False
        h_at_ball = b_h / 0.15                 # measured ball:player ratio
        ground_y = coef[0] * h_at_ball + coef[1]
        # Image y grows downward, so "above the ground" is a SMALLER y.
        return (ground_y - by) > BALL_AIRBORNE_BH * h_at_ball

    raw_on = {}
    airborne_frames = 0
    for fr, players in per_frame.items():
        if fr not in ball_at:
            continue
        bx, by = ball_at[fr][0], ball_at[fr][1]
        b_h = ball_at[fr][3]
        if _airborne(players, bx, by, b_h):
            raw_on[fr] = None       # in flight — nobody has it
            airborne_frames += 1
            continue
        best, best_rank = None, None
        for p in players:
            # Depth check first: if the ball's apparent size is wildly out of
            # proportion to this player's, the two are at different distances
            # and the overlap is a projection artefact, not possession.
            if b_h > 0 and p["h"] > 0:
                ratio = b_h / p["h"]
                if not (BALL_DEPTH_RATIO_MIN <= ratio <= BALL_DEPTH_RATIO_MAX):
                    continue
            x0, y0, x1, y1 = _box(p)
            # Distance to the box: zero inside it, else to the nearest edge.
            dx = max(x0 - bx, 0.0, bx - x1)
            dy = max(y0 - by, 0.0, by - y1)
            d = float(np.hypot(dx, dy))
            inside = (dx == 0.0 and dy == 0.0)
            if not inside and d > ON_BALL_RADIUS_BH * max(p["h"], 0.02):
                continue            # too far to be in possession at all
            # Containment outranks any distance. Among several containing boxes,
            # or several near ones, the tie-break is how deep or how close.
            if inside:
                depth = min(bx - x0, x1 - bx, by - y0, y1 - by)
                rank = (0, -depth)  # deeper inside wins
            else:
                rank = (1, d)
            if best_rank is None or rank < best_rank:
                best, best_rank = p["id"], rank
        raw_on[fr] = best
    # HYSTERESIS, not a symmetric majority vote.
    #
    # The vote was over +/- ON_BALL_SMOOTH_S = 0.40s, which at 5fps is two
    # samples — far too short to settle anything, and symmetric, so it gives the
    # incumbent no advantage over a neighbour who happens to win one frame.
    # Result: basketball produced 22 possession spells in 30s with a median of
    # 1.07s and EIGHT under half a second. Real possession lasts seconds; the
    # marker was flickering between adjacent players rather than following one.
    #
    # Possession is a state, so model it as one: the holder keeps the ball until
    # a challenger has been the per-frame pick for a sustained stretch. That is
    # asymmetric on purpose — taking the ball off someone should need more
    # evidence than keeping it.
    on_ball = {}
    order_fr = sorted(raw_on)
    current, run_id, run_len = None, None, 0
    need = max(1, int(ON_BALL_STICK_S * src_fps))
    for fr in order_fr:
        pick = raw_on[fr]
        if pick == run_id:
            run_len += 1
        else:
            run_id, run_len = pick, 1
        # An incumbent who is no longer in the frame cannot still have the ball.
        # Without this the marker simply vanishes: `current` keeps naming a track
        # that has ended, nothing in the frame matches the id, and nobody is
        # drawn. It cost football 25 on-ball frames at stick=0.6 and got worse
        # the stickier it went, which is the opposite of what stickiness is for.
        if current is not None and not any(p["id"] == current
                                           for p in per_frame.get(fr, [])):
            current = None
        # Claim the ball when nobody holds it, or when a challenger has held the
        # per-frame pick long enough to have earned it.
        if current is None or (run_id != current and run_len >= need):
            if run_id is not None or run_len >= need:
                current = run_id
        on_ball[fr] = current

    # MINIMUM POSSESSION, added 5 Sep. A spell shorter than this is not
    # possession, it is the ball passing by - a header, a deflection, a
    # clearance. The user's words on the amateur clip: "sometimes possession is
    # just headbutting the ball away from you".
    #
    # The measurement that backs it: possession spells per clip, before this
    # rule. Nobody in amateur EVER held the ball for more than 1.83s, against
    # 4.47s on allstars and 5.17s on basketball, and 7 of its 23 spells were
    # under half a second. A clip whose longest possession is under two seconds
    # is a clip in which possession mostly is not happening.
    #
    # Applied AFTER the hysteresis rather than inside it, because the two rules
    # answer different questions: stickiness decides who wins a contested
    # frame, this decides whether the resulting spell was ever real. Cleared to
    # None rather than reassigned - the honest answer for a loose ball is that
    # nobody has it, not that the nearest player does.
    if MIN_POSSESSION_S > 0 and order_fr:
        need_len = max(1, int(round(MIN_POSSESSION_S * src_fps)))
        start = 0
        for i in range(1, len(order_fr) + 1):
            at_end = i == len(order_fr)
            if at_end or on_ball[order_fr[i]] != on_ball[order_fr[start]]:
                who = on_ball[order_fr[start]]
                if who is not None and (i - start) < need_len:
                    for j in range(start, i):
                        on_ball[order_fr[j]] = None
                start = i

    out_frames = []
    for fr in range(total):
        ps = per_frame.get(fr, [])
        for p in ps:
            p["on_ball"] = (on_ball.get(fr) == p["id"])
        b = ball_at.get(fr)
        out_frames.append({"frame": fr, "players": ps,
                           "ball": {"x": round(b[0], 5), "y": round(b[1], 5),
                                    "w": round(b[2], 5), "h": round(b[3], 5)} if b else None})

    return {
        "clip": data["clip"], "model": data["model"], "tag": data["tag"],
        "fps_sampled": data["fps"], "source_fps": src_fps,
        "n_frames": total,
        "teams": {k: v for k, v in team_map.items()},
        "accents": accents,
        "kit_counts": dict(kit_counts.most_common()),
        "identities": display,
        "cuts": cuts,
        "degenerate_frames": degenerate,
        "ball_outliers": ball_outliers,
        "camera_drift": [round(cam_cum[0], 4), round(cam_cum[1], 4)],
        "camera_track": [[f, round(v[0], 4), round(v[1], 4)]
                         for f, v in sorted(cam_at.items())],
        "diag": {k: ([round(float(np.percentile(v, p)), 3) for p in (10, 50, 90)]
                     if v else []) for k, v in diag.items()},
        "frames": out_frames,
    }, finished, cuts, kit_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detections", type=Path)
    ap.add_argument("--no-accel-gate", action="store_true",
                    help="disable the acceleration gate; for the A/B that "
                         "justifies it, and for measuring raw detection jitter")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--coast", type=float, default=None,
                    help="override MAX_COAST_S: how long a track survives unseen")
    ap.add_argument("--stick", type=float, default=None,
                    help="override ON_BALL_STICK_S: how long a challenger must "
                         "be the per-frame pick before possession transfers. "
                         "Higher = steadier but laggier on genuine turnovers")
    ap.add_argument("--ball-min-conf", type=float, default=None,
                    help="reject ball detections below this confidence. OFF by "
                         "default. Targets decoys the kinematic filters cannot "
                         "see — a spot mark or boot beside the real ball moves "
                         "plausibly. Try 0.82")
    ap.add_argument("--reconfirm", action="store_true",
                    help="a lone sighting between two absences is not drawn")
    ap.add_argument("--dump-costs", type=Path, default=None,
                    help="DEBUG. Write the full association cost matrix — every "
                         "(track, candidate) pair with its distance, gate, kit "
                         "multiplier, number bonus, height term and final cost — "
                         "as JSONL. Off unless given")
    ap.add_argument("--dump-costs-until", type=int, default=10 ** 9,
                    help="only dump costs for source frames <= this")
    args = ap.parse_args()
    ACCEL_GATE_ON[0] = not args.no_accel_gate
    if args.dump_costs is not None:
        COST_LOG["on"] = True
        COST_LOG["max_frame"] = args.dump_costs_until
    if args.coast is not None:
        globals()["MAX_COAST_S"] = args.coast
    if args.ball_min_conf is not None:
        globals()["BALL_MIN_CONF"] = args.ball_min_conf
    if args.stick is not None:
        globals()["ON_BALL_STICK_S"] = args.stick
    RECONFIRM[0] = args.reconfirm

    if not args.detections.exists():
        sys.exit(f"no such file: {args.detections}")
    data = json.loads(args.detections.read_text(encoding="utf-8"))

    result, tracks, cuts, kit_counts = run(data, args.debug)

    if args.dump_costs is not None:
        args.dump_costs.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_costs.open("w", encoding="utf-8") as fh:
            for r in COST_LOG["rows"]:
                fh.write(json.dumps(r) + "\n")
        # Join back to what was drawn via the "track" field on every
        # player-frame in the tracks json.
        print(f"  cost dump       {len(COST_LOG['rows'])} rows "
              f"-> {args.dump_costs}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (args.detections.stem + "__tracks.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    read = sum(1 for v in result["identities"].values()
               if v["number_source"] == "read")
    drawn = [len(f["players"]) for f in result["frames"]]
    with_ball = sum(1 for f in result["frames"] if f["ball"])
    print(f"\n  kit colours     {dict(list(kit_counts.most_common(6)))}")
    print(f"  teams           {result['teams']}  (top two by sightings)")
    print(f"  raw tracks      {len(tracks)}")
    print(f"  identities      {len(result['identities'])}  "
          f"({read} numbered by reading, "
          f"{len(result['identities']) - read} by fallback id)")
    print(f"  camera cuts     {len(cuts)}"
          + (f"  at frames {[c['frame'] for c in cuts][:12]}" if cuts else ""))
    deg = result.get("degenerate_frames", [])
    if deg:
        shown = ", ".join("{}({}->{})".format(x["frame"], x["raw"], x["unique"])
                          for x in deg[:6])
        print(f"  degenerate      {len(deg)} frames the model repeated itself "
              f"on, skipped: {shown}")
    dg = result.get("diag", {})
    if dg.get("rate"):
        print(f"  match rate      p10 {dg['rate'][0]}  p50 {dg['rate'][1]}  "
              f"p90 {dg['rate'][2]}   (share of tracks that found a detection)")
        print(f"  births/frame    p10 {dg['births'][0]}  p50 {dg['births'][1]}  "
              f"p90 {dg['births'][2]}")
        print(f"  unmatched why   kit-vetoed p50 {dg['kit_blocked'][1]}  "
              f"nothing-in-gate p50 {dg['gate_missed'][1]}")
    if ACCEL_REJECTS:
        # Deduped by (frame, track). The raw list counts CANDIDATE PAIRS, and the
        # cost matrix scores every track against every detection in gate — so
        # most entries are a track declining some other player's box, which it
        # would have declined on distance anyway. The number that means anything
        # is how many track-frames were affected at all.
        events = {(r["frame"], r["track"]) for r in ACCEL_REJECTS}
        worst = sorted(ACCEL_REJECTS, key=lambda r: -r["resid"])[:4]
        print(f"  residual gate   {len(events)} track-frames had a candidate "
              f"refused over {MAX_RESIDUAL_BH} body heights "
              f"({len(ACCEL_REJECTS)} pairs scored)")
        for r in worst:
            print(f"                  t+{(r['frame'] or 0)/30:5.2f}s track "
                  f"{r['track']:<4} residual={r['resid']:5.2f} bh  "
                  f"jump {r['dist']:.4f}")
    bo = result.get("ball_outliers", [])
    if bo:
        kinds = Counter(o["kind"] for o in bo)
        print(f"  ball rejected   {len(bo)}  {dict(kinds)}")
    print(f"  players drawn   min {min(drawn)} med "
          f"{sorted(drawn)[len(drawn)//2]} max {max(drawn)}  per output frame")
    print(f"  ball drawn      {with_ball}/{result['n_frames']} frames")
    if args.debug:
        for tr in sorted(tracks, key=lambda t: -len(t.obs))[:12]:
            print(f"    track {tr.id:>3}  {len(tr.obs):>4} obs  kit={tr.kit:<8} "
                  f"num={tr.number}  votes={dict(tr.num_votes)}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
