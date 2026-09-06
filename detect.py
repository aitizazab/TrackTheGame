"""
Stage 1 of the pipeline: video in, per-frame detections out.

    clip.mp4 --ffmpeg--> N sampled frames --parallel VLM calls--> detections.json

Everything downstream (tracking, interpolation, rendering) consumes the JSON this
writes and never touches the video again.

WHAT THE MEASUREMENTS DECIDED, so the constants below are arguable rather than
arbitrary (see docs/budget_probe.jsonl):

  Native resolution by default. Latency is FLAT against image size — 320px cost
  1.77s to first data, 1280px cost 1.96s across a 14x byte range. Downscaling
  buys nothing and throws away the pixels a football lives in. --width exists
  only so resolution can be swept as an ablation.

  max_tokens=1600. Luna spends its whole allowance on reasoning before emitting
  anything: at max_tokens=128, all 128 were reasoning tokens and the content was
  empty. Set this too low and you get HTTP 200 with no answer in it.

  Timeout, and stragglers dropped. Concurrency is effectively free — median call
  time stayed flat at ~3.9s all the way to N=64 — but every batch contains one
  call that hangs for 4-25s, and wall clock equals that straggler every time.
  A deadline is therefore the ONLY latency control that binds. A dropped frame
  costs a little accuracy; a straggler costs the whole budget.

  Fractional coordinates. Invariant under the resolution sweep, and 'fraction
  mode' was the winning format for this model on the previous project.

  Kit colour as a word, never "team A". Each call is independent, so a call
  asked for "team A" picks its own A and the teams shuffle between frames.
  Colour is observer-independent; the colour->team mapping happens once, later.

  Scratchpad field FIRST in the schema. JSON emits fields in schema order, so a
  coordinate field placed first must be produced with zero tokens spent thinking.

    uv run detect.py clips/football_wide.mp4 --fps 10
    uv run detect.py clips/football_wide.mp4 --fps 10 --width 768 --tag res768
    uv run detect.py clips/football_wide.mp4 --frames 0,30,60   # cheap smoke test
"""

import argparse
import base64
import copy
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# The pre-D26 prompt and schemas, frozen byte-identical from git. Only the
# --arms path uses them; nothing here changes the shipping configuration.
from prompt_v1 import (COMPACT_ORDER_V1, COMPACT_SCHEMA_V1, PROMPT_V1,
                       SCHEMA_V1)
from prompt_v3 import (COMPACT_ORDER_V3, COMPACT_SCHEMA_V3, COORD_SPACE_V3,
                       PROMPT_V3, SCHEMA_V3)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
# CHANGED 5 Sep. It was openai/gpt-5.6-luna, which is (a) no longer the
# shipping model and (b) no longer listed in the provider catalogue at all - a
# run without --model now 404s on all 150 calls with "No endpoints found".
# Free, since nothing is billed, but it wastes a full wall clock and writes an
# empty detections file. The default should be the thing we actually ship.
DEFAULT_MODEL = "google/gemini-3.7-flash"
LOG = Path("docs/run_log.jsonl")
DETECTIONS = Path("outputs/detections")

# Measured, not guessed. On the first real run Luna's reasoning came in at 516,
# 1034 and 1306 tokens on three consecutive frames. At a 1600 cap the two long
# ones left 294 and 566 tokens for content and the JSON was cut off mid-string.
# The earlier estimate of ~260 reasoning tokens came from the output probe, which
# used a one-line prompt on a synthetic image — it does not survive contact with
# a real crowded frame. 4000 covers ~2000 reasoning plus ~25 players of content.
# It is a ceiling, not a reservation: frame 300 only billed 1315.
#
# 4000 was still not enough. Over 300 real frames, reasoning ran to 2578 tokens
# at the top end and two frames truncated anyway. Raised to 6500. This is not a
# number to keep nudging — reasoning is 64% of all output tokens on this model,
# which makes reasoning effort (ablation A5) the real lever, not the cap.
# LOWERED 5 Sep, 6500 -> 4000. It is the reservation the provider holds against
# the in-flight budget, and an oversized one directly worsens the 402 ceiling
# that stopped a run on 1 Sep. Measured need: compact v2 output is ~1750 median
# and the largest reasoning seen on real football is 2578; the only run that
# ever truncated was 3.8 + the long v1 prompt, at 6236-6243. D4's warning still
# applies - a cap measured on easy input is not a cap - so this keeps ~1.5x
# headroom over the worst REAL v2/v3 frame rather than trimming to the median.
MAX_TOKENS = 4000

# ---------------------------------------------------- coordinate conventions
#
# ONE CONVENTION PER MODEL, PINNED HERE, NEVER INFERRED AT RUNTIME.
#
# Models do not reliably obey the coordinate convention the prompt and schema
# ask for. `google/gemini-3.1-flash-lite` returned pixels — x to 986, y to 765
# on a 1280x720 frame — against a schema that says "fraction of image width,
# 0.0-1.0" with a worked example. It is not an error and not a parse failure:
# pixels are perfectly valid numbers, so the schema passes, the tracker runs,
# and every marker lands off-screen.
#
# The tempting fix is a detector that sniffs each response and adapts. The last
# project did that and lost days to it — an adaptive reader cannot tell a model
# that switched convention mid-run from one that is simply wrong this frame, and
# every ambiguity becomes a new special case. So: a model is pinned or it does
# not run. Adding one is a deliberate, cheap, human step:
#
#     uv run detect.py <clip> --model <new/model> --probe-convention
#
# which sends three frames, reports what came back, and tells you what line to
# add. Nothing adapts silently, and a run always records which convention it
# used so a result can never be misread later.
# EVERY ENTRY MUST BE OBSERVED. Not inferred from the family, not assumed from a
# sibling, not carried over from a previous version. `gpt-5.6-luna-pro` was added
# here on "same family as luna" reasoning and removed again — that is exactly the
# guessing this table exists to prevent, and a wrong pin is worse than no pin,
# because no pin refuses to run while a wrong one silently produces garbage.
# THOUSANDTH exists because the probe originally had only two buckets and the
# world has three. Gemini's documented box format is a 0-1000 normalised space,
# and both 3.x flash-lites use it. The probe classified "> 1.5" as PIXEL, so a
# 0-1000 coordinate was silently mapped through the wrong divisor: x squashed to
# 78% of the frame, y stretched 39% past the bottom. The renders were wrong and
# the JSON looked fine.
#
# The tell was in the data the whole time: x+w topped out at 1003 and 993 for two
# independent models — right at 1000, not near 1280 — while y+h reached 835 and
# 865, which is *impossible* in a 720-pixel frame. A ceiling that lands on 1000
# in both axes is a normalised space, not a resolution.
FRACTION, PIXEL, THOUSANDTH = "fraction", "pixel", "thousandth"
COORD_CONVENTION = {
    "openai/gpt-5.6-luna":             FRACTION,  # observed, 300 frames
    "qwen/qwen3-vl-32b-instruct":      FRACTION,  # observed, 89 frames
    "qwen/qwen3-vl-30b-a3b-instruct":  FRACTION,  # probed, max coord 0.8
    "google/gemini-2.5-flash-lite":    FRACTION,  # observed, 73 frames
    "google/gemini-3.1-flash-lite":    THOUSANDTH,  # x+w tops out at 1003
    "google/gemini-3.5-flash-lite":    THOUSANDTH,  # x+w tops out at 993
    "google/gemini-3.7-flash":         FRACTION,  # probed, max coord 1.0
    "google/gemini-3.8-flash":         FRACTION,  # probed 4 Sep, max coord 1.0
    "mistralai/mistral-large-2512":    FRACTION,  # probed, max coord 0.9
}
# A pattern, recorded but NOT acted on: the Gemini 3.x *flash-lite* models both
# return pixels, while 2.5-flash-lite and 3.7-flash return fractions. So the
# convention tracks the lite line rather than the version number. Tempting to
# extrapolate to the next lite release. Do not — probe it. The whole value of
# this table is that every row was measured, and one inferred row poisons the
# guarantee for all of them.
# A real wall-clock deadline (see call_one). 43s sits just above the p90 of 41.6s
# measured on the first full run, so it keeps almost every genuine call and cuts
# only the tail that would otherwise own the whole batch. Dropped frames cost a
# little accuracy; a straggler costs the entire budget, and the tracker
# interpolates across the gap either way.
# 43.0 until 2 Sep, chosen when p90 was 41.6s. Lowered to 35.0 because wall clock
# IS the slowest call (measured: wall minus slowest call is 0.7-1.2s across three
# runs, which is the ffmpeg extract), so the deadline is the only direct cap on
# it. On a good run nothing changes — flex_30s peaked at 30.3s, so neither 43 nor
# 35 ever fires — but on a bad one it bounds the damage: the 2 Sep run peaked at
# 43.2s and had calls sitting on the old limit.
#
# Do NOT drop it to 25 to "hit the target". The latency distribution has a hard
# shoulder, not a thin tail: 28s costs 3.3% of frames, 25s costs 32.7%.
TIMEOUT_S = 35.0
# DYNAMIC STRAGGLER CUT. Once this share of calls has returned, the rest get
# CUT_GRACE_S more and are then abandoned mid-stream.
#
# NOT a lower TIMEOUT_S, which is a guess at where the tail will land and costs
# 32.7% of frames on a bad day. This adapts to the run: it only starts the clock
# once most of the work is already in.
#
# Swept across all four full-clip runs. Wall saved / frames lost of 150:
#
#   p99   0.8-4.4s, but cuts SAVES NOTHING - its slow calls already sit on the
#         35s deadline, so the p99 threshold lands above them        1-3 frames
#   p97   2.5 / 3.7 / 5.3 / 14.9s                                    4 frames
#   p95   3.1 / 4.1 / 6.3 / 17.9s                                    7 frames
#   p93   3.3 / 4.4 / 6.5 / 18.1s                                   10 frames
#   p90   3.5 / 4.7 / 7.0 / 18.8s, and amateur's blind spell hits    14 frames
#         0.60s, the exact coast limit, where tracks start dying
#
# Returns flatten after p97 while frame loss grows linearly: p97->p95 buys
# 0.4-3.0s for three more frames, p95->p93 buys 0.3s for three more. 0.97 takes
# every clip under ~21s, costs 2.7% of frames, and its worst blind spell is
# 0.40s - half the tracker's 0.60s coast.
#
# Accuracy cost measured by replaying the cut offline: identities +0 to +2,
# match rate unchanged to -0.003, marker-frames -1.7% to -2.3%.
CUT_SHARE = 0.97
CUT_GRACE_S = 1.5
# Set to a perf_counter deadline once the share is reached; call_one checks it
# between chunks so an abandoned call actually closes its connection rather than
# running on in a thread the process must still join at exit.
CUT_AT = [None]
SOURCE_FPS = 30           # fetch_clips.py normalises every clip to this

# ---------------------------------------------------------------- the schema

# Field order is load-bearing. `scene` first gives the model somewhere to think
# before it has to commit to a number. Everything is required and
# additionalProperties is false because strict structured-output modes demand it.
SCHEMA = {
    "name": "frame_detections",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene", "players", "ball"],
        "properties": {
            "scene": {
                "type": "string",
                "description": "One short sentence: camera framing, lighting, "
                               "and the shirt colour of each team. Written "
                               "BEFORE any coordinates, as working-out."
            },
            # KITS REMOVED 3 Sep. The field worked - accent was non-null on
            # all but 95 of ~10k frames - but its only consumer is the
            # renderer's dE >= 30 fallback, which has NEVER fired on any clip,
            # including the one the user nominated as the similar-kit case.
            # With no accents the fallback degrades from two tiers to one and
            # still returns a distinguishable colour via opposite(). Cost was a
            # prompt paragraph, a schema subtree, ~10 output tokens a frame and
            # one more judgement call per frame. track.py and render.py already
            # tolerate the field being absent, so restoring this block is the
            # whole of the undo - do that first if clip 5 has similar kits.
            "players": {
                "type": "array",
                "description": "One entry per player on the field of play, "
                               "GOALKEEPERS INCLUDED. Box tightly: top edge at "
                               "the crown of the head, bottom edge where the "
                               "feet meet the ground. Empty array is valid and "
                               "correct if none are visible.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x", "y", "w", "h", "kit", "num", "role"],
                    "properties": {
                        "x": {"type": "number",
                              "description": "LEFT edge of the player's box, "
                                             "fraction of image width, 0.0-1.0"},
                        "y": {"type": "number",
                              "description": "TOP edge of the player's box, "
                                             "fraction of image height, 0.0-1.0"},
                        "w": {"type": "number",
                              "description": "box width, fraction of image width"},
                        "h": {"type": "number",
                              "description": "box height, fraction of image height"},
                        "kit": {"type": "string",
                                "description": "Shirt colour as one common "
                                               "word. Judge the colour itself; "
                                               "never \"team A\" or \"home\" - "
                                               "each frame is judged on its own "
                                               "and the words must agree across "
                                               "frames."},
                        "num": {"type": ["integer", "null"],
                                "description": "Jersey number ONLY if you can "
                                               "actually read it. null otherwise. A wrong number is far worse than no number."},
                        # Without this a goalkeeper in a third kit colour is
                        # indistinguishable from an unstable colour word, and the
                        # renderer cannot tell "green kit = keeper" from "someone
                        # said green once by mistake".
                        "role": {"type": "string", "enum": ["outfield", "goalkeeper"],
                                 "description": "goalkeeper if they wear a "
                                                "different kit from both teams "
                                                "and stand in/near a goal. In "
                                                "sports with no goalkeeper, "
                                                "every player is outfield"}
                    }
                }
            },
            "ball": {
                "type": ["object", "null"],
                "description": "null when the ball is not visible, which it "
                               "often is not. The ball is ABOVE the playing "
                               "surface: not a mark painted on it, and not "
                               "something a player is wearing or carrying. "
                               "Never place it where you think it ought to be.",
                "additionalProperties": False,
                # TRIED AND REMOVED 3 Sep: a `kind` field naming which sport's
                # ball this is, so the clip's sport could be set by majority vote
                # and minority reports rejected as misidentifications. On
                # basketball all 136 detections said "basketball" — including
                # every one the geometric filters rejected — so the wrong-sport
                # filter never fired. The model names the sport it is watching,
                # not the object: by the time it fills the field it has already
                # decided "this is the ball", so the field sits downstream of the
                # error rather than checking it. Cost 6.8% in tokens for nothing.
                #
                # The clothing line in the prompt, added at the same time, DID
                # work — see the note there.
                "required": ["x", "y", "w", "h", "conf"],
                "properties": {
                    "x": {"type": "number", "description": "left edge, fraction"},
                    "y": {"type": "number", "description": "top edge, fraction"},
                    "w": {"type": "number", "description": "width, fraction"},
                    "h": {"type": "number", "description": "height, fraction"},
                    "conf": {"type": "number"}
                }
            }
        }
    }
}

# PROMPT REWRITTEN 3 Sep. Every rule used to be stated twice - once here in
# prose and once in a schema `description` - and "scene first" three times,
# though only the schema's property order actually binds. Roughly 900 of 1981
# prompt tokens were a second copy.
#
# The schema now carries the whole per-field contract; this text carries only
# what a schema cannot say - who is NOT a player, and how to count. Also gone:
# the worked coordinate example (constrained generation already fixes the shape)
# and the claim that a long box makes "the marker float below their feet". That
# rationale was RETRACTED when the box render showed top and bottom edges
# correct, and leaving it in was telling the model its boxes run long.
#
# The eleven named ball decoys - boot, sock, glove, shinpad, bandage, sleeve,
# centre spot, penalty spot, arcs, lines, logos - collapse to one positive test
# in the ball description. Naming a distractor inside a negation raises its
# salience, and the clothing line moved basketball decoys only 4 -> 2 while
# football_cuts kept plenty.
#
# Judgement calls per frame: 11 -> 7. That is the target, not word count -
# reasoning tokens scale with the number of decisions, not the length of the
# instructions.
PROMPT = """One frame of sports footage. Report every player on the playing
surface, and the ball.

Do NOT report: referees and other match officials, substitutes and anyone on the
bench, coaches, medical staff, the crowd, ball boys.

Report players who are partly hidden or partly out of frame, and box only the
part you can actually see. Report exactly as many players as you can see - there
is no expected number and it does not depend on the sport."""


# ---------------------------------------------------------- frame extraction

def need(tool: str) -> None:
    if not shutil.which(tool):
        sys.exit(f"{tool} not found on PATH. Restart your shell if newly installed.")


def source_frame_count(clip: Path) -> int:
    """How many frames the finished video will have.

    The tracker renders every source frame, so it needs the true total. Deriving
    it from the last SAMPLED index truncates the tail: at 10fps from 30fps the
    last sample is frame 897, and 898-899 would never be drawn. Clips are
    normalised to CFR by fetch_clips.py, so duration x fps is exact.
    """
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True)
    try:
        return int(round(float(r.stdout.strip()) * SOURCE_FPS))
    except (ValueError, TypeError):
        return 0


def extract_frames(clip: Path, fps: int, out_dir: Path) -> list:
    """Sample `fps` frames per second. Returns [(source_frame_index, Path), ...].

    The source index matters downstream: the tracker needs real elapsed time
    between observations, and the renderer has to put annotations back on the
    right frame of the original 900. Sampling at 10fps from 30fps means sampled
    frame k came from source frame k*3 — record it rather than recompute it
    somewhere else and risk the two drifting apart.
    """
    need("ffmpeg")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"fps={fps}",
         "-q:v", "2", str(out_dir / "%05d.jpg")],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{r.stderr[-800:]}")
    step = SOURCE_FPS / fps
    return [(round(i * step), p)
            for i, p in enumerate(sorted(out_dir.glob("*.jpg")))]


def draw_ruler(img):
    """Overlay a coordinate reference on the frame (ablation A1).

    The generalised form of the ruler trick from Ball Detector: a VLM has no
    metric readout, so asking "where is this" is a weak spatial regression. Put
    tick marks and numerals IN the pixels and it becomes reading text, which is
    the thing these models are best at.

    Drawn as an OVERLAY, not as an added border. Adding a band would change the
    image dimensions and therefore what a fraction means, silently shifting
    every coordinate — the arm would then be measuring two things at once.

    The hypothesis here is not about accuracy. The user's read is that the model
    can already find players; it cannot read their shirts. So the thing to watch
    is whether handing it the coordinate system reduces REASONING tokens, which
    are most of the bill.
    """
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arialbd.ttf", max(11, w // 90))
    except OSError:
        font = ImageFont.load_default()
    for i in range(1, 10):
        f = i / 10.0
        x, y = int(w * f), int(h * f)
        # top edge: vertical ticks, labelled with the x fraction
        d.rectangle([x - 1, 0, x + 1, int(h * 0.022)], fill=(0, 0, 0, 190))
        d.text((x + 3, 1), f".{i}", font=font, fill=(255, 255, 0, 235),
               stroke_width=2, stroke_fill=(0, 0, 0, 220))
        # left edge: horizontal ticks, labelled with the y fraction
        d.rectangle([0, y - 1, int(w * 0.013), y + 1], fill=(0, 0, 0, 190))
        d.text((2, y + 2), f".{i}", font=font, fill=(255, 255, 0, 235),
               stroke_width=2, stroke_fill=(0, 0, 0, 220))
    return img


def draw_grid(img, div: int = 10):
    """Overlay a thin full-frame grid (ablation A1b).

    DIFFERENT HYPOTHESIS FROM `draw_ruler`, which is kept beside it because A1's
    rejection is a measured result and must stay reproducible. The ruler put
    numerals on the EDGES: to use it the model still has to project a player
    inward from the margin, which is the same spatial regression it was already
    bad at, and it measurably drove reasoning tokens UP.

    A grid puts a reference line WITHIN a body-width of every player, so
    localising becomes "which cell, and where inside it" — a local judgement
    against a visible landmark instead of a global one against a distant scale.
    The target here is not jersey numbers, it is the ~0.05 frame-fraction
    localisation JITTER that produces ring fly-outs, which the effort probe
    showed is stochastic and therefore not fixable by prompting or reasoning.

    Magenta because nothing on a football pitch is magenta: not grass, not any
    kit colour seen so far (blue, white, orange, red, yellow, green, black), and
    not the ball. A grey or white grid risks being read as a painted line, and
    the prompt already tells the model painted markings are not the ball.

    Thin and translucent on purpose. A player is ~20px wide and ~80px tall at
    1080p, so a 1px line at a third opacity cannot hide one, and the lines sit
    at fixed screen positions so they are trivially separable from anything that
    moves. Note the cost: fine lines are exactly the high-frequency detail JPEG
    spends bits on, so bytes-per-frame will rise — measured in the run summary.
    """
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arialbd.ttf", max(11, w // 100))
    except OSError:
        font = ImageFont.load_default()
    # MINOR lines at half spacing, unlabelled and fainter. Without them the
    # nearest reference can be 0.05 away, which is the same magnitude as the
    # localisation jitter this arm exists to test - a landmark no closer than
    # the error it is meant to remove is not a landmark. With them, nothing on
    # the frame is more than 0.025 from a line, about half a player's width.
    for i in range(1, div * 2):
        if i % 2 == 0:
            continue
        f = i / (div * 2)
        x, y = int(w * f), int(h * f)
        d.line([(x, 0), (x, h)], fill=(255, 0, 255, 38), width=1)
        d.line([(0, y), (w, y)], fill=(255, 0, 255, 38), width=1)
    for i in range(1, div):
        f = i / div
        x, y = int(w * f), int(h * f)
        d.line([(x, 0), (x, h)], fill=(255, 0, 255, 70), width=1)
        d.line([(0, y), (w, y)], fill=(255, 0, 255, 70), width=1)
    # Edge numerals, so the lines carry values rather than only structure.
    for i in range(1, div):
        f = i / div
        x, y = int(w * f), int(h * f)
        d.text((x + 2, 1), f".{i}", font=font, fill=(255, 255, 0, 235),
               stroke_width=2, stroke_fill=(0, 0, 0, 220))
        d.text((2, y + 1), f".{i}", font=font, fill=(255, 255, 0, 235),
               stroke_width=2, stroke_fill=(0, 0, 0, 220))
    return img


GRID_NOTE = """
A thin magenta GRID is drawn over the image: brighter lines every 0.1 of width
and height, fainter ones halfway between them at every 0.05, with yellow
numerals on the top and left edges. It is an overlay, not part of the scene: do
not report a grid line as a player or as the ball. Use the nearest lines to
place each box - read off which cell a player stands in and where in that cell
their feet are."""


RULER_NOTE = """
A COORDINATE RULER is drawn on the image. Yellow numerals along the top edge
mark x = .1 to .9; along the left edge they mark y = .1 to .9. Read positions
off it rather than estimating them. The ruler itself is not part of the scene —
do not report it as a player or as the ball."""


def encode(path: Path, width: int = None, ruler: bool = False,
           grid: bool = False) -> tuple:
    img = Image.open(path).convert("RGB")
    if width and width != img.width:
        img = img.resize((width, round(width * img.height / img.width)))
    if ruler:
        img = draw_ruler(img)
    if grid:
        img = draw_grid(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    raw = buf.getvalue()
    return ("data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"),
            len(raw), img.width, img.height)


# --------------------------------------------------------------- the caller

def to_fractions(result: dict, convention: str, img_w: int, img_h: int) -> dict:
    """Rewrite one frame's boxes into fractions, per the model's PINNED
    convention. Applied unconditionally for that model — it never inspects the
    values to decide, because that is the guessing this is designed to avoid."""
    if convention == FRACTION:
        return result
    if convention == THOUSANDTH:
        sx = sy = 1000.0        # a resolution-independent normalised space
    else:
        sx, sy = float(img_w), float(img_h)
    for p in result.get("players") or []:
        p["x"] /= sx; p["w"] /= sx
        p["y"] /= sy; p["h"] /= sy
    b = result.get("ball")
    if b:
        b["x"] /= sx; b["w"] /= sx
        b["y"] /= sy; b["h"] /= sy
    return result


COORD_MIN, COORD_MAX = -0.10, 1.50
FRAME_REJECT_SHARE = 0.40

# A player box wider than this many times its own height is not a player.
#
# MEASURED IN PIXEL SPACE, AND THAT DISTINCTION IS THE WHOLE POINT. Fractions
# divide x,w by the frame WIDTH and y,h by the frame HEIGHT, so on 16:9 a
# fraction-space w/h understates the true aspect by exactly 16/9 = 1.778. The
# same trap is already recorded in HANDOFF §5: "box aspect looked fine in
# fraction space because a 16:9 frame inflates it by 1.78x". The earlier
# proposal for this guard — reject fraction w/h > 1.0 — is a PIXEL aspect of
# 1.78, and would throw away real players.
#
# 36,329 player boxes from shipping-config runs (gemini-3.7-flash, fraction,
# effort not low), pixel aspect: p50 0.463, p90 0.635, p99 0.874, p99.9 1.276.
# Above 1.4 there are 22 boxes, and they split into two populations with an
# empty band between them:
#
#   plausible, 19 boxes, 1.40-1.94  e.g. 92x50px, kit orange, allstars f828,
#                                   identical in two independent runs. Sprawled
#                                   or diving players. A diving goalkeeper is
#                                   ~2.5 (the user's figure) and belongs here.
#   corrupt, 3 boxes, 5.61-8.44     e.g. 1075x127px — a ribbon across most of
#                                   the frame. The renderer draws its ring at
#                                   w x 1.9, i.e. 2043px on a 1280px frame.
#
# 3.0 sits in that empty band: 1.5x above the widest plausible real box, 1.9x
# below the narrowest corrupt one, and comfortably clear of a 2.5 diving keeper.
# It rejects 3 boxes in 36,329 — 0.008% — and every one of them is a ribbon.
# Not tuned to a target rejection rate; placed in a gap the data actually has.
MAX_PIXEL_ASPECT = 3.0


def validate_boxes(result: dict, frame_aspect: float = 16 / 9) -> dict:
    """Discard boxes that are not on the 0..1 scale. Never reinterpret them.

    `gemini-3.7-flash` is pinned FRACTION and mostly obeys, but on 1080p input it
    intermittently emits 0-1000 or pixel values — 0.5% of frames at medium
    effort, 36% at low, and 13 frames in one run mixed BOTH SCALES INSIDE A
    SINGLE RESPONSE. No pin can fix a response that has no single convention.

    The distinction that keeps this honest: a box with x=945 against a schema
    that says 0..1 is INVALID, not ambiguous. We are not guessing what it meant
    and rescaling it — that is the adaptive-catcher trap. We are throwing it
    away, exactly as we would a malformed number.

    A frame that loses more than FRAME_REJECT_SHARE of its boxes is failed
    outright, because what remains is not a view of the pitch — same reasoning
    as the degenerate-repetition check in track.py. The tracker coasts across it.

    The second test is shape: a box on the 0..1 scale can still be a ribbon
    across the frame, and one such box drives a ring wider than the video. See
    MAX_PIXEL_ASPECT — it is a PIXEL aspect, so `frame_aspect` must be the
    detected image's w/h, not assumed.
    """
    players = result.get("players") or []
    total = len(players)
    ok = []
    wide = 0
    for p in players:
        vals = (p.get("x"), p.get("y"),
                (p.get("x") or 0) + (p.get("w") or 0),
                (p.get("y") or 0) + (p.get("h") or 0))
        if not (all(v is not None and COORD_MIN <= v <= COORD_MAX for v in vals)
                and (p.get("w") or 0) > 0 and (p.get("h") or 0) > 0):
            continue
        if (p["w"] / p["h"]) * frame_aspect > MAX_PIXEL_ASPECT:
            wide += 1
            continue
        ok.append(p)
    dropped = total - len(ok)

    b = result.get("ball")
    ball_dropped = False
    if b:
        vals = (b.get("x"), b.get("y"),
                (b.get("x") or 0) + (b.get("w") or 0),
                (b.get("y") or 0) + (b.get("h") or 0))
        if not all(v is not None and COORD_MIN <= v <= COORD_MAX for v in vals):
            result["ball"] = None
            ball_dropped = True

    if not dropped and not ball_dropped:
        return {}
    result["players"] = ok
    # `wide` is counted inside `dropped` deliberately: a frame that is mostly
    # ribbons is as corrupt as one that is mostly off-scale, and should fail the
    # same way. It is reported separately so the run log can tell the two apart.
    return {"total": total, "dropped": dropped, "wide": wide,
            "ball_dropped": ball_dropped,
            "frame_rejected": total > 0 and dropped / total > FRAME_REJECT_SHARE}


def report_convention(records, model):
    """Probe output: say what the model did, and the exact line to pin it with."""
    vals = [v for r in records if r.get("ok")
            for p in (r["result"].get("players") or [])
            for v in (p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"])]
    if not vals:
        print("\n  no usable detections — cannot determine convention")
        return
    hi = max(vals)
    # Three buckets, and the boundaries are deliberately wide with a refusal in
    # between rather than a nearest-match. A two-bucket version silently swallowed
    # 0-1000 as "pixels" and produced wrong renders that looked fine in JSON.
    if hi <= 1.5:
        guess = FRACTION
    elif 300 <= hi <= 1010:
        guess = THOUSANDTH
    elif hi > 1010:
        guess = PIXEL
    else:
        print(f"\n  largest coordinate seen: {hi:.1f}")
        print(f"  -> AMBIGUOUS. Between 1.5 and 300 fits no convention we know")
        print(f"     (percent? a 0-255 space? a small frame?). Do not pin this;")
        print(f"     send the number to be looked at.")
        return
    if guess == PIXEL and hi < 1.5 * 1010:
        print(f"\n  largest coordinate seen: {hi:.1f}")
        print(f"  -> looks like PIXELS but is close to the 0-1000 boundary.")
        print(f"     Check whether y+h ever exceeds the frame HEIGHT — if it")
        print(f"     does, it cannot be pixels. Do not pin this yet.")
        return
    print(f"\n  largest coordinate seen: {hi:.1f}")
    print(f"  -> this model reports {guess.upper()}"
          f"{'S' if guess != THOUSANDTH else ' (0-1000 normalised)'}")
    print(f"\n  Add to COORD_CONVENTION in detect.py:\n")
    print(f'      "{model}": {guess.upper()},')
    print(f"\n  Nothing was pinned automatically. Add the line, then re-run.")


# ------------------------------------------------------- ablation variants
#
# Every switch below defaults OFF so the control run is byte-identical to what
# shipped. Each is measured separately against that control on the same clip.

# The per-field rules currently live in BOTH the prompt and the schema's
# `description` fields. --container removes them from the prompt and leaves them
# only in the schema, where each sits adjacent to the field being generated.
CONTAINER_PROMPT = """One frame of sports footage. Report the players and the ball.

  - Players on the field of play, GOALKEEPERS INCLUDED.
  - Not referees, substitutes, coaches, the crowd or ball boys.
  - Every player you can see, partly hidden ones included at a low conf.
  - Do not pad the list to a round number. Seven visible means seven reported.
  - Painted markings are not the ball. If you cannot see it, ball is null."""

# --compact replaces the per-player object with a fixed-order array. The key
# names are over half the bytes of each player record and there are ~18 of them
# per frame, and output is ~82% of the bill.
COMPACT_ORDER = ["x", "y", "w", "h", "kit", "num", "role"]
COMPACT_SCHEMA = {
    "name": "frame_detections_compact",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene", "players", "ball"],
        "properties": {
            # DESCRIPTIONS RESTORED 3 Sep, and the reason first given for it
            # was wrong. The table originally here paired A7 (--terse-schema, 31
            # Aug, OBJECT form, different clip, output +11%) with basketball
            # v2-vs-rich (3 Sep, ARRAY form, output +6%) and read a single
            # mechanism across both. They are different experiments pointing
            # opposite ways, and v2-vs-rich was not controlled either: v2 ran six
            # hours before the ball `kind` field was dropped, so it was
            # "stripped + kind" against "described - kind", two changes worth
            # ~6.8% in opposite directions.
            #
            # The one controlled pair (rich vs nc, identical 1981 prompt tokens,
            # same clip, one hour apart) says the format is the whole saving and
            # the descriptions are unmeasured:
            #
            #   object + descriptions   reasoning 1345   content 864   $0.004944
            #   array  + descriptions   reasoning 1361   content 387   $0.004031
            #
            # Reasoning moves 1.2%, which is noise; content halves. So "a model
            # given less guidance reasons longer" has nothing behind it. What the
            # split does show is where the money is: reasoning is 77.9% of output
            # tokens and about 64% of the whole per-video bill, and every schema
            # change so far has been aimed at the other 18%.
            "scene": {"type": "string",
                      "description": "One short sentence: camera framing, "
                                     "lighting, and the shirt colour of each "
                                     "team. Written BEFORE any coordinates, as "
                                     "working-out."},
            # KITS REMOVED 3 Sep. The field worked - accent was non-null on
            # all but 95 of ~10k frames - but its only consumer is the
            # renderer's dE >= 30 fallback, which has NEVER fired on any clip,
            # including the one the user nominated as the similar-kit case.
            # With no accents the fallback degrades from two tiers to one and
            # still returns a distinguishable colour via opposite(). Cost was a
            # prompt paragraph, a schema subtree, ~10 output tokens a frame and
            # one more judgement call per frame. track.py and render.py already
            # tolerate the field being absent, so restoring this block is the
            # whole of the undo - do that first if clip 5 has similar kits.
            "players": {
                "type": "array",
                # The array form loses per-field typing entirely: `items` has to
                # admit number, string and null, so nothing stops position 0
                # being a string or position 5 a float. In the object form `x`
                # was constrained to number and `num` to integer|null. This
                # sentence is now the ONLY thing carrying that contract, which is
                # why it states the type of every position as well as its meaning.
                "description": ("One array per player, GOALKEEPERS INCLUDED, "
                                "ALWAYS in this order: "
                                "[x, y, w, h, kit, num, role]. "
                                "x = LEFT edge of the box, fraction of image "
                                "width, 0.0-1.0 (number). "
                                "y = TOP edge of the box, fraction of image "
                                "height, 0.0-1.0 (number). "
                                "w = box width as a fraction of image width "
                                "(number). "
                                "h = box height as a fraction of image height "
                                "(number). "
                                "Box tightly: top edge at the crown of the head, "
                                "bottom edge where the feet meet the ground. "
                                "kit = shirt colour as one ordinary word "
                                "(string): red, blue, white, yellow, green, "
                                "black, orange, purple. Judge the colour itself; "
                                "never \"team A\" or \"home\", because each frame "
                                "is judged on its own and the words must agree "
                                "across frames. "
                                "num = the number on the shirt as an integer, "
                                "ONLY if you can genuinely read it, otherwise "
                                "null. A wrong number is far worse than no "
                                "number. "
                                "role = \"goalkeeper\" if they wear a kit unlike "
                                "both teams and stand in or near a goal, "
                                "otherwise \"outfield\"; in sports with no "
                                "goalkeeper every player is \"outfield\". "
                                "An empty array is valid and correct if no "
                                "players are visible."),
                "items": {"type": "array",
                          "items": {"type": ["number", "string", "null"]}}},
            "ball": {"type": ["array", "null"],
                     "description": ("[x, y, w, h, conf] or null, boxed tightly, "
                                     "all as fractions 0.0-1.0. x = left edge, "
                                     "y = top edge, w = width, h = height, "
                                     "conf = 0.0 to 1.0. The ball is ABOVE the "
                                     "playing surface: not a mark painted on it, "
                                     "and not something a player is wearing or "
                                     "carrying. If you cannot see the ball, null "
                                     "- null is common and correct. Never place "
                                     "it where you think it ought to be."),
                     "items": {"type": ["number", "null"]}},
        }}}


# ------------------------------------------------------- prompt versions
#
# Two complete prompt+schema packages, selectable per arm. They are a PACKAGE
# and not two independent knobs: D26 changed the prose and the schema together,
# because removing player `conf` from the schema while leaving the prose that
# explains how to calibrate it would test neither version of anything. The
# report must therefore name the variable as "the D26 prompt+schema package",
# not "prompt length".
#
#   v1  pre-D26. 3291-character prompt, every rule stated twice, player `conf`,
#       the kits/accent block, eleven named ball decoys. Produced every number
#       in the project before 3 Sep, including the signed-off video.
#   v2  the D26 rewrite. 440 characters, rules stated once, no player `conf`,
#       no kits/accent, one positive test for the ball. NEVER RUN.
PROMPT_SETS = {
    "v1": {"prompt": PROMPT_V1, "schema": SCHEMA_V1,
           "compact_schema": COMPACT_SCHEMA_V1, "compact_order": COMPACT_ORDER_V1,
           "player_conf": True, "coords": None},
    "v2": {"prompt": None, "schema": None,           # filled in below; the live
           "compact_schema": None, "compact_order": None,   # module globals ARE v2
           "player_conf": False, "coords": None},
    # v3 asks for 0-1000 INTEGERS, so it carries its own coordinate space and
    # overrides the per-model pin. `coords` is None for v1/v2, meaning "use
    # COORD_CONVENTION[model]" exactly as before.
    "v3": {"prompt": PROMPT_V3, "schema": SCHEMA_V3,
           "compact_schema": COMPACT_SCHEMA_V3, "compact_order": COMPACT_ORDER_V3,
           "player_conf": False, "coords": COORD_SPACE_V3},
}
PROMPT_SETS["v2"].update(prompt=PROMPT, schema=SCHEMA,
                         compact_schema=COMPACT_SCHEMA,
                         compact_order=COMPACT_ORDER)
DEFAULT_PROMPT_VERSION = "v2"


# ---------------------------------------------------------------------- v4
#
# v2, minus `role`, plus the frame-edge exclusion. FRACTIONS ARE KEPT.
#
# v3 bundled three changes and one of them was poison: asking for 0-1000
# INTEGERS made the model emit ~6% of frames on a wrong scale entirely - 63
# boxes with h > 0.25 where v1 and v2 produced ZERO, max height 0.773 against
# 0.108. Because the whole frame shifts together, the frame median shifts too,
# so neither the 0..1 range test nor a relative-height guard can see it: only
# 1 of 973 boxes exceeded 4x its own frame's median. It renders as giant rings
# and nothing warns you. Measured saving was 6.4% of cost. Not worth it.
#
# That the bundle hid which change did the damage is my error, and the reason
# v4 changes exactly two things, both of which are inert or prose:
#   - `role` removed: measured to have 0 references in track.py and render.py.
#   - the frame-edge line: a prompt sentence, testable by watching for fly-ins.
def _v4_schema(src, compact):
    import copy as _copy
    sc = _copy.deepcopy(src)
    props = sc["schema"]["properties"]["players"]
    if compact:
        props["description"] = (props["description"]
                                .replace("[x, y, w, h, kit, num, role]",
                                         "[x, y, w, h, kit, num]"))
        i = props["description"].find("role = ")
        j = props["description"].find("An empty array is valid")
        if i > 0 and j > i:
            props["description"] = props["description"][:i] + props["description"][j:]
    else:
        it = props["items"]
        it["required"] = [k for k in it["required"] if k != "role"]
        it["properties"].pop("role", None)
    return sc


PROMPT_V4 = """One frame of sports footage. Report every player on the playing
surface, and the ball.

Do NOT report: referees and other match officials, substitutes and anyone on the
bench, coaches, medical staff, the crowd, ball boys.

Do NOT report a player who is more than half outside the frame. If you can see
most of them, report them and box only the part you can actually see. Report
exactly as many players as you can see - there is no expected number and it does
not depend on the sport."""

PROMPT_SETS["v4"] = {
    "prompt": PROMPT_V4,
    "schema": _v4_schema(SCHEMA, False),
    "compact_schema": _v4_schema(COMPACT_SCHEMA, True),
    "compact_order": [k for k in COMPACT_ORDER if k != "role"],
    "player_conf": False,
    "coords": None,
}


# ---------------------------------------------------------------------- v5
#
# v4, with the ball returned as a LIST OF CANDIDATES instead of one object.
#
# WHY THIS MIGHT BE EXPENSIVE, which is the whole point of testing it before
# building on it. "Report the ball" lets the search TERMINATE: once the model
# has found something it believes is the ball, it has answered the question.
# "Report every object that could be the ball" cannot terminate early - it is
# only answerable by examining the whole frame. If the model was already doing
# an exhaustive pass and simply discarding the alternatives, exposing them is
# nearly free. If it was stopping at the first hit, this makes every frame a
# full search and the reasoning cost could rise sharply.
#
# Those two possibilities are indistinguishable from the outside, which is why
# this is an arm run and not a schema change.
#
# The wording deliberately does NOT say "one is enough if you are sure" - that
# would restore the early exit and test nothing.
def _v5_schema(src, compact):
    import copy as _copy
    sc = _copy.deepcopy(src)
    props = sc["schema"]["properties"]
    order = [k for k in props if k != "ball"] + ["balls"]
    if compact:
        props["balls"] = {
            "type": "array",
            "description": ("Every object in this frame that could plausibly be "
                            "the ball, MOST LIKELY FIRST, at most three. Each is "
                            "[x, y, w, h, conf] with x, y, w, h as fractions "
                            "0.0-1.0 and conf 0.0-1.0. Include the ones you "
                            "reject as well as the one you believe: a pale "
                            "round thing that turned out to be a boot, a sock, "
                            "a painted mark or a logo still belongs here, with "
                            "a low conf. An empty array is valid and correct "
                            "when nothing in the frame could be the ball."),
            "items": {"type": "array", "items": {"type": ["number", "null"]}}}
    else:
        props["balls"] = {
            "type": "array",
            "description": ("Every object that could plausibly be the ball, most "
                            "likely first, at most three. Include rejected "
                            "candidates with a low conf."),
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["x", "y", "w", "h", "conf"],
                      "properties": {
                          "x": {"type": "number"}, "y": {"type": "number"},
                          "w": {"type": "number"}, "h": {"type": "number"},
                          "conf": {"type": "number"}}}}
    props.pop("ball", None)
    sc["schema"]["properties"] = {k: props[k] for k in order}
    sc["schema"]["required"] = order
    return sc


PROMPT_V5 = PROMPT_V4 + """

For the ball, report EVERY object in the frame that could plausibly be one,
most likely first. Include the ones you decide against, with a low confidence -
a pale round shape that turns out to be a boot, a sock, a painted marking or a
logo is still a candidate. Do not stop at the first one you find."""

PROMPT_SETS["v5"] = {
    "prompt": PROMPT_V5,
    "schema": _v5_schema(PROMPT_SETS["v4"]["schema"], False),
    "compact_schema": _v5_schema(PROMPT_SETS["v4"]["compact_schema"], True),
    "compact_order": list(PROMPT_SETS["v4"]["compact_order"]),
    "player_conf": False,
    "coords": None,
}



# ---------------------------------------------------------------------- v6
#
# v4, plus a BALL VISIBILITY judgement emitted BEFORE the ball coordinates.
#
# The hypothesis, from the decoy timestamps the user identified by watching:
# every single one occurred while the real ball was OCCLUDED. A boot, a sock,
# an advertising board and a painted spot were each reported as the ball at a
# moment the ball itself could not be seen. So the failure is not "picked the
# wrong object", it is "would not return nothing".
#
# Field ORDER is the whole mechanism, not decoration. JSON emits in schema
# order, so `ball_state` is produced before any ball coordinate exists - the
# model has to commit to "hidden" while it still costs nothing, rather than
# rationalise a box it has already written. Same reasoning that puts `scene`
# first (A4).
#
# The risk is D17: a model asked to classify will classify confidently whether
# or not it knows. If v6 reports "clear" on the frames we know are occluded,
# that is the answer and this route is closed too.
def _v6_schema(src, compact):
    import copy as _copy
    sc = _copy.deepcopy(src)
    props = sc["schema"]["properties"]
    props["ball_state"] = {
        "type": "string",
        "enum": ["clear", "partly_hidden", "hidden"],
        "description": ("Before giving any ball coordinates, say whether you "
                        "can actually see the ball. clear = plainly visible. "
                        "partly_hidden = you can see part of it. hidden = you "
                        "cannot see it, because a player is in the way, it is "
                        "out of frame, or it is simply not there. Decide this "
                        "FIRST and answer honestly; hidden is a common and "
                        "correct answer.")}
    order = [k for k in props if k not in ("ball", "ball_state")]
    order += ["ball_state", "ball"]
    sc["schema"]["properties"] = {k: props[k] for k in order}
    sc["schema"]["required"] = order
    return sc


PROMPT_V6 = PROMPT_V4 + """

Before reporting the ball, say whether you can see it: clear, partly_hidden, or
hidden. If it is hidden - blocked by a player, out of frame, or not there - say
hidden and set ball to null. Do NOT substitute the nearest pale round object: a
boot, a sock, a glove, a painted marking or a logo is not the ball, and "I
cannot see it" is a correct and common answer."""

PROMPT_SETS["v6"] = {
    "prompt": PROMPT_V6,
    "schema": _v6_schema(PROMPT_SETS["v4"]["schema"], False),
    "compact_schema": _v6_schema(PROMPT_SETS["v4"]["compact_schema"], True),
    "compact_order": list(PROMPT_SETS["v4"]["compact_order"]),
    "player_conf": False,
    "coords": None,
}



def normalise_result(result: dict, compact: bool,
                     version: str = DEFAULT_PROMPT_VERSION) -> dict:
    """Turn a compact array response back into the standard dict shape.

    Everything downstream - the validator, the tracker, the renderer - keeps
    working on one format. The wire format is an ablation; the internal one is
    not, and letting a switch leak past this function would mean testing the
    output format and the whole pipeline at the same time.

    That is also why player `conf` is SYNTHESISED here rather than deleted from
    the tracker. The model stopped reporting it on 3 Sep: across 80,374 recorded
    detections only 333 - 0.41% - ever came back below ByteTrack's 0.50 split,
    so it cost a judgement call and an output element per player for a decision
    it never actually made. track.py reads it in four places, and giving every
    player the modal value (1.0, which 24,134 detections reported outright)
    keeps those paths on one format: the ByteTrack low-confidence pass becomes
    explicitly empty, Kalman measurement noise uniform, and the jersey vote an
    unweighted count.

    Ball `conf` is REAL and still reported. It multiplies the ball speed gate,
    where decoys sit at median 0.68 against 0.95 for real balls - the only
    signal that separates a static decoy from a slow ball.
    """
    # v1 reports a real player `conf`; v2 does not and it is synthesised. The
    # version has to be threaded here rather than inferred from the row length,
    # because a v2 row that happens to arrive with a trailing element would
    # otherwise be silently reinterpreted as a v1 row.
    spec = PROMPT_SETS[version]
    order = spec["compact_order"]
    if not compact:
        for pl in result.get("players") or []:
            if isinstance(pl, dict) and not spec["player_conf"]:
                pl["conf"] = 1.0
        return result
    out = []
    malformed = []
    for row in result.get("players") or []:
        if not isinstance(row, list) or len(row) < len(order):
            # COUNTED, NOT SILENT, since 4 Sep. This branch discarded a player
            # with no counter, no log line and no error, which meant "the model
            # did not report this player" and "the model reported them in a
            # shape we could not read" were indistinguishable downstream - and
            # the second is exactly the failure the compact array format makes
            # possible, because `items` admits number|string|null at every
            # position and nothing constrains the row LENGTH.
            malformed.append(row if isinstance(row, list) else type(row).__name__)
            continue
        pl = dict(zip(order, row))
        for k in ("x", "y", "w", "h"):
            try:
                pl[k] = float(pl[k])
            except (TypeError, ValueError):
                pl[k] = 0.0
        try:
            pl["num"] = int(pl["num"]) if pl["num"] is not None else None
        except (TypeError, ValueError):
            pl["num"] = None
        pl["kit"] = str(pl.get("kit") or "")
        if spec["player_conf"]:
            try:
                pl["conf"] = float(pl["conf"])
            except (TypeError, ValueError):
                pl["conf"] = 1.0
        else:
            pl["conf"] = 1.0
        out.append(pl)
    result["players"] = out
    if malformed:
        result["_malformed_rows"] = malformed[:5]
        result["_malformed_count"] = len(malformed)
    # v5 returns a LIST of ball candidates. Collapse it to the single `ball`
    # everything downstream expects - the top candidate - while keeping the
    # whole list under `ball_candidates`. Nothing reads the list yet: the point
    # of the first run is to find out what asking for it COSTS, not to build on
    # it. Keeping the collapse here means track.py and render.py are untouched
    # and the arms stay comparable.
    if "balls" in result:
        cands = result.pop("balls") or []
        norm = []
        for c in cands:
            if isinstance(c, list) and len(c) >= 5:
                norm.append({"x": float(c[0]), "y": float(c[1]),
                             "w": float(c[2]), "h": float(c[3]),
                             "conf": float(c[4])})
            elif isinstance(c, dict) and "x" in c:
                norm.append(c)
        result["ball_candidates"] = norm
        result["ball"] = norm[0] if norm else None
        return result
    b = result.get("ball")
    if isinstance(b, list) and len(b) >= 5:
        result["ball"] = {"x": float(b[0]), "y": float(b[1]), "w": float(b[2]),
                          "h": float(b[3]), "conf": float(b[4])}
    elif isinstance(b, list):
        result["ball"] = None
    return result


def build_messages(prompt: str, data_url: str, system: bool, image_first: bool):
    """Where the instructions sit, and in what order relative to the image.

    --system moves them to a system message: a cleaner cache prefix, and some
    models weight system content more heavily.
    --image-first puts the picture before the text, so the instructions are the
    most recent thing in context when generation begins.
    """
    img = {"type": "image_url", "image_url": {"url": data_url}}
    txt = {"type": "text", "text": prompt}
    if system:
        return [{"role": "system", "content": prompt},
                {"role": "user", "content": [img]}]
    return [{"role": "user", "content": [img, txt] if image_first else [txt, img]}]


def build_schema(scene_last: bool = False, terse: bool = False,
                 compact: bool = False,
                 version: str = DEFAULT_PROMPT_VERSION) -> dict:
    """SCHEMA with the two ablation knobs applied (A4 and A7).

    Field ORDER is semantically load-bearing, not cosmetic: JSON emits fields in
    schema order, so moving `scene` to the end means the model must commit to
    every coordinate before writing a word of working-out. That is the whole
    hypothesis of A4, which is why this rebuilds the dict rather than mutating
    a shared one.
    """
    spec = PROMPT_SETS[version]
    s = copy.deepcopy(spec["compact_schema"] if compact else spec["schema"])
    if terse:
        def strip(node):
            if isinstance(node, dict):
                node.pop("description", None)
                for v in node.values():
                    strip(v)
            elif isinstance(node, list):
                for v in node:
                    strip(v)
        strip(s["schema"])
    if scene_last:
        props = s["schema"]["properties"]
        # Derived from the schema rather than hardcoded: v1 also has "kits", and
        # a literal ["players","ball","scene"] silently DELETED it, producing a
        # v1 run with no kit block and no error anywhere.
        order = [k for k in props if k != "scene"] + ["scene"]
        s["schema"]["properties"] = {k: props[k] for k in order}
        s["schema"]["required"] = order
        s["name"] = "frame_detections_scene_last"
    return s


def call_with_retry(session, headers, model, frame_idx, path, width, timeout,
                    schema, effort, convention, variant=None, tries=3,
                    version=DEFAULT_PROMPT_VERSION):
    """Retry connection-level failures. Do NOT retry a deadline.

    A 1080p JPEG is roughly three times the bytes of a 720p one, and 300 of them
    uploading at once produced 73 ConnectionErrors and an SSLError in a single
    run — 25% of the clip simply missing. Those are transport failures, not the
    model declining, and they succeed on a second attempt.

    A deadline is different: it means the model really did take longer than we
    are willing to wait, and retrying would just spend the budget twice for the
    same answer. Dropped-on-deadline stays dropped.
    """
    last = None
    waited = 0.0
    for attempt in range(tries):
        rec = call_one(session, headers, model, frame_idx, path, width, timeout,
                       schema, effort, convention, variant, version)
        if rec.get("ok"):
            if attempt:
                rec["retries"] = attempt
                rec["retry_wait_s"] = round(waited, 2)
            return rec
        err = rec.get("error") or ""
        transient = ("ConnectionError" in err or "SSLError" in err
                     or "ChunkedEncoding" in err or "RemoteDisconnected" in err
                     # Both are 200s that carried no usable completion, found on
                     # 2 Sep across basketball and football_cuts. Neither is a
                     # config problem and neither was being retried: one crashed
                     # as KeyError, the other advised raising a cap that was
                     # already twice what the run used. 3 frames of 300 lost to
                     # provider hiccups that a second attempt would have fixed.
                     or "no choices in a 200" in err
                     or "provider generated 0 tokens" in err)
        # HTTP 429 is the canonical retryable error and was NOT being retried:
        # on 2 Sep, 49 of 150 calls died to it without a single second attempt,
        # while 29 transport failures were recovered by this same function.
        #
        # It needs a DELAY, unlike a transport failure. The refusal comes from
        # Google's shared upstream quota, so retrying instantly just asks the
        # same overloaded endpoint again. And because we fire every frame at
        # once, undelayed retries would arrive as one synchronised wave — a
        # thundering herd that re-triggers the limiter it is waiting on. The
        # jitter is what breaks the wave up; the doubling is what backs off.
        # 402 means the key is out of credit. Every remaining frame will fail
        # the same way, and retrying is guaranteed waste - HANDOFF item 4 asked
        # for this after six of eight losses on 1 Sep were silently dropped
        # rate limits. Raise so the run stops instead of grinding through 150
        # doomed calls and writing a detections file full of holes.
        # A 404 means the model id does not resolve - a typo, or a model that
        # has been delisted. Every remaining call fails identically, so stop
        # rather than grind through the whole clip. Same reasoning as 402.
        if rec.get("status") == 404:
            raise SystemExit(
                "\nHTTP 404 - the model id did not resolve. Run ABORTED.\n"
                f"  {(rec.get('error') or '')[:200]}\n")
        if rec.get("status") == 402:
            raise SystemExit(
                "\nHTTP 402 - the key is out of credit. Run ABORTED so the "
                "rest of the frames are not spent failing.\n"
                f"  provider said: {(rec.get('error') or '')[:180]}\n")
        rate_limited = rec.get("status") == 429
        if not (transient or rate_limited):
            return rec
        last = rec
        if rate_limited and attempt < tries - 1:
            # 1s, 2s, 4s, each x0.75-1.33. Worst case ~9.3s added, inside a 35s
            # deadline. Recorded separately so it can never be mistaken for
            # model latency: t0 is set inside call_one, AFTER this sleep, so
            # latency_s is unaffected and only wall clock absorbs the wait.
            delay = (2 ** attempt) * random.uniform(0.75, 1.333)
            waited += delay
            time.sleep(delay)
    if last is not None:
        last["retries"] = tries - 1
        if waited:
            last["retry_wait_s"] = round(waited, 2)
    return last


def call_one(session, headers, model, frame_idx, path, width, timeout,
             schema, effort, convention, variant=None,
             version=DEFAULT_PROMPT_VERSION):
    v = variant or {}
    # SECTION TIMING. `latency_s` starts after this and ends when the last byte
    # arrives, so encode and base64 were invisible to every latency number this
    # project has ever quoted. On volleyball the wall clock was 36.5s while the
    # slowest call reported 25.6s and none of the gap was retries, backoff,
    # HTTP status or payload size - which is not answerable without splitting
    # the call into the parts that can actually be slow.
    t_enc0 = time.perf_counter()
    data_url, nbytes, w, h = encode(path, width, v.get("ruler", False),
                                    v.get("grid", False))
    t_encode = time.perf_counter() - t_enc0
    prompt = (CONTAINER_PROMPT if v.get("container")
              else PROMPT_SETS[version]["prompt"])
    if v.get("ruler"):
        prompt = prompt + "\n" + RULER_NOTE
    if v.get("grid"):
        prompt = prompt + "\n" + GRID_NOTE
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
        "response_format": {"type": "json_schema", "json_schema": schema},
        "messages": build_messages(prompt, data_url, v.get("system", False),
                                   v.get("image_first", False)),
    }
    # allow_fallbacks=false is the point, not a detail. With fallbacks on, a pin
    # is a preference and the run can still be served — and billed — by another
    # tier, which is the failure this flag exists to make impossible.
    if v.get("provider_order"):
        body["provider"] = {"order": v["provider_order"], "allow_fallbacks": False}
    if effort:
        body["reasoning"] = {"effort": effort}
    t0 = time.perf_counter()
    # `model` and `prompt_version` are recorded PER CALL, not once per run. In an
    # interleaved arm run they differ between calls, and the log writer merges
    # the record last so these win over the run-level defaults.
    rec = {"frame": frame_idx, "w": w, "h": h, "bytes": nbytes,
           "model": model, "prompt_version": version,
           "t_encode_s": round(t_encode, 3)}
    try:
        # `timeout=` in requests is NOT a wall-clock limit. It is a socket
        # inactivity limit: the clock resets every time any byte arrives. A
        # provider that trickles keepalives can therefore hold a connection open
        # indefinitely without ever tripping it — which is exactly what happened
        # on the first full run, where the nominal timeout was 20s and one call
        # ran 168.9 seconds and set the wall clock for all 300.
        #
        # Streaming the response lets us check a real deadline between chunks.
        # The first element is the CONNECT timeout, and it governs the socket
        # while the request body is still going out. It was 10s, chosen
        # arbitrarily when streaming was added, and it turned out to be the
        # thing that broke 1080p:
        #
        #   85 of 90 failures were
        #     ConnectionError('Connection aborted.', TimeoutError('The write
        #     operation timed out'))
        #   at a median of 12.3s, against successful calls averaging 25.1s.
        #
        # A 313KB upload contending with 299 siblings cannot finish inside 10
        # seconds, so the socket was killed mid-send — while the model was
        # perfectly willing to answer. The same code at 720p (165KB, 49MB total)
        # had ZERO transport failures, which is the control.
        #
        # The wall-clock deadline is enforced separately in the read loop below,
        # so this only needs to be generous enough to get the bytes out.
        t_req0 = time.perf_counter()
        r = session.post(ENDPOINT, headers=headers, json=body,
                         timeout=(min(30.0, timeout), timeout), stream=True)
        # post() with stream=True returns once the response HEADERS arrive, so
        # this is connect + upload + time-to-first-byte: the model thinking.
        rec["t_ttfb_s"] = round(time.perf_counter() - t_req0, 3)
        t_stream0 = time.perf_counter()
        rec["status"] = r.status_code
        if r.status_code != 200:
            rec["ok"] = False
            rec["latency_s"] = time.perf_counter() - t0
            rec["error"] = r.text[:200]
            return rec
        parts = []
        for chunk in r.iter_content(16384):
            now = time.perf_counter()
            if now - t0 > timeout:
                r.close()
                rec["ok"] = False
                rec["latency_s"] = now - t0
                rec["error"] = "deadline"
                return rec
            if CUT_AT[0] is not None and now > CUT_AT[0]:
                r.close()
                rec["ok"] = False
                rec["latency_s"] = now - t0
                rec["error"] = "straggler cut"
                return rec
            parts.append(chunk)
        rec["latency_s"] = time.perf_counter() - t0
        rec["t_stream_s"] = round(time.perf_counter() - t_stream0, 3)
        t_parse0 = time.perf_counter()
        payload = json.loads(b"".join(parts).decode("utf-8", "replace"))
        usage = payload.get("usage") or {}
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        det = usage.get("completion_tokens_details") or {}
        rec["reasoning_tokens"] = det.get("reasoning_tokens")
        # Deliverable 2 asks for cost against speed against approach. We had been
        # measuring tokens and inferring price from a figure carried over from the
        # last project. OpenRouter returns the real charge per call — record it.
        rec["cost_usd"] = usage.get("cost")
        # Which endpoint actually answered. Without this, a tier change is only
        # visible by solving backwards from cost_usd, which is how the 28->31 Aug
        # flex/standard drift went unnoticed for three days.
        rec["provider"] = payload.get("provider")
        # A 200 does NOT guarantee a completion. Basketball frame 660 came back
        # 200 with no "choices" key at all, which crashed as KeyError: 'choices'
        # and was reported as a code bug rather than as the provider hiccup it
        # is. Name it, and let call_with_retry treat it as transient.
        choices = payload.get("choices") or []
        if not choices:
            rec["ok"] = False
            rec["error"] = f"no choices in a 200 response (keys: {sorted(payload)[:6]})"
            return rec
        choice = choices[0]
        content = (choice.get("message") or {}).get("content") or ""
        # Name the failure precisely. Truncation arrives as a JSONDecodeError
        # about an unterminated string, which reads like a schema or parsing
        # problem and sends you looking in the wrong place. finish_reason says
        # what actually happened.
        if choice.get("finish_reason") == "length":
            rec["ok"] = False
            rec["error"] = (f"truncated at max_tokens — reasoning used "
                            f"{rec.get('reasoning_tokens')}, raise MAX_TOKENS")
            return rec
        if not content.strip():
            # Two different failures wore this one message until 2 Sep, and the
            # advice it gave was wrong for one of them:
            #
            #   completion_tokens > 0  -> the budget really did go on reasoning,
            #       which is what the output probe predicted. Raise MAX_TOKENS.
            #   completion_tokens == 0 -> the model generated NOTHING. Raising
            #       the cap cannot help; it is a null response and it succeeds
            #       on a retry.
            #
            # Basketball frame 870 and cuts frame 300 were both the second kind
            # (out=0, rsn=0), and were being told to raise a cap the run was
            # using less than half of.
            rec["ok"] = False
            if (rec.get("completion_tokens") or 0) == 0:
                rec["error"] = "empty response: provider generated 0 tokens"
            else:
                rec["error"] = (f"empty content, {rec.get('completion_tokens')} "
                                f"tokens all spent on reasoning — raise MAX_TOKENS")
            return rec
        rec["result"] = normalise_result(json.loads(content),
                                         v.get("compact", False), version)
        # Lifted out of the result so it reaches the run log without ending up
        # in the detections file the tracker reads.
        n_bad = rec["result"].pop("_malformed_count", 0)
        if n_bad:
            # `_malformed_rows` is a sample capped at 5; the count is the truth.
            rec["malformed_count"] = n_bad
            rec["malformed_rows"] = rec["result"].pop("_malformed_rows", None)
        if convention:
            rec["result"] = to_fractions(rec["result"], convention, w, h)
        bad = validate_boxes(rec["result"], frame_aspect=(w / h if h else 16 / 9))
        if bad:
            rec["invalid_boxes"] = bad
            if bad.get("frame_rejected"):
                rec["ok"] = False
                rec["error"] = (
                    f"coordinate corruption: {bad['dropped']} of "
                    f"{bad['total']} boxes invalid "
                    f"({bad['dropped'] - bad['wide']} off the 0..1 scale, "
                    f"{bad['wide']} wider than {MAX_PIXEL_ASPECT}:1)")
                return rec
        rec["t_parse_s"] = round(time.perf_counter() - t_parse0, 3)
        rec["ok"] = True
        return rec
    except requests.exceptions.Timeout:
        rec["ok"] = False
        rec["latency_s"] = time.perf_counter() - t0
        rec["error"] = "timeout"      # expected and fine; the tracker fills the gap
        return rec
    except Exception as e:
        rec["ok"] = False
        rec["latency_s"] = time.perf_counter() - t0
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec


# Wall-clock accounting, filled in as the run proceeds so the summary can say
# where the time actually went rather than leaving it to be inferred from a
# single `wall` number.
T_EXTRACT = [0.0]
T_WRITE = [0.0]


def run_pool(fns, workers):
    """Run every call, then abandon the slowest few once most have landed.

    Arms the cut only after CUT_SHARE of the calls have returned, so a slow
    provider day shifts the deadline instead of costing frames. Results come
    back in SUBMISSION order, not completion order, because everything
    downstream keys on frame index.
    """
    from concurrent.futures import as_completed
    CUT_AT[0] = None
    out = [None] * len(fns)
    pool_t0 = time.perf_counter()

    def stamped(fn):
        """Wrap a call so the run log can separate WAITING from CALLING.

        `latency_s` starts inside call_one, after the frame is encoded and just
        before the POST - so everything before that is invisible to it. On the
        volleyball run the wall clock was 36.5s while the slowest call reported
        25.6s, and none of the 11s difference was retries, backoff, HTTP errors,
        encoding or payload size. These two offsets say whether a thread started
        late (scheduling / GIL contention on encode and base64) or finished late
        (teardown after the straggler cut closed its socket).
        """
        def go():
            t_in = time.perf_counter() - pool_t0
            rec = fn()
            if isinstance(rec, dict):
                rec["t_start_s"] = round(t_in, 2)
                rec["t_done_s"] = round(time.perf_counter() - pool_t0, 2)
            return rec
        return go

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(stamped(fn)): i for i, fn in enumerate(fns)}
        need = max(1, int(round(CUT_SHARE * len(fns))))
        done = 0
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
            done += 1
            if done >= need and CUT_AT[0] is None and done < len(fns):
                CUT_AT[0] = time.perf_counter() + CUT_GRACE_S
    CUT_AT[0] = None
    return out


def report_sections(records, wall):
    """Where the wall clock went, by section.

    Every number here is per call except the extract/write rows, so the columns
    do NOT sum to the wall clock - 150 calls run concurrently. Read `worst` as
    "the slowest single call spent this long in this phase", because the wall is
    set by one call, not by the average of them.
    """
    def col(key):
        v = sorted(r[key] for r in records if r.get(key) is not None)
        return v or [0.0]

    def line(name, v, note=""):
        n = len(v)
        print(f"    {name:<14}{v[n // 2]:8.2f}{v[int(n * 0.9)]:9.2f}"
              f"{v[-1]:9.2f}   {note}")

    print(f"  --- where the time went ------------------------------------")
    print(f"    {'section':<14}{'p50':>8}{'p90':>9}{'worst':>9}")
    print(f"    {'ffmpeg extract':<14}{T_EXTRACT[0]:8.2f}{'':>9}{'':>9}   "
          f"once, before any call")
    line("encode+b64", col("t_encode_s"), "JPEG + base64, NOT in latency_s")
    line("connect+TTFB", col("t_ttfb_s"), "upload + the model thinking")
    line("stream body", col("t_stream_s"), "response download")
    line("parse+validate", col("t_parse_s"), "json + normalise + box checks")
    line("latency_s", col("latency_s"), "TTFB + stream, what we have quoted")
    st = col("t_start_s")
    line("thread start", st, "queue delay before the call began")
    dn = col("t_done_s")
    line("thread done", dn, "offset from pool start")
    print(f"    {'write json':<14}{T_WRITE[0]:8.2f}{'':>9}{'':>9}")
    slow = max(records, key=lambda r: r.get("t_done_s") or 0, default=None)
    if slow and slow.get("t_done_s"):
        print(f"    the call that SET the wall: frame {slow.get('frame')}, "
              f"started {slow.get('t_start_s')}s in, "
              f"encode {slow.get('t_encode_s')}s, ttfb {slow.get('t_ttfb_s')}s, "
              f"stream {slow.get('t_stream_s')}s, done {slow.get('t_done_s')}s")
        unexplained = wall - (slow.get("t_done_s") or 0)
        print(f"    wall minus that call's completion: {unexplained:.2f}s "
              f"(pool teardown + write)")


def parse_arms(spec: str, args) -> list:
    """`model:version,model:version,...` -> a list of arm dicts.

    Every arm is validated up front, before a single call is made, because the
    failure mode this guards against is spending two thirds of a run's money and
    then dying on the third arm's unpinned model.
    """
    arms = []
    for n, part in enumerate(p.strip() for p in spec.split(",") if p.strip()):
        bits = part.split(":")
        model = bits[0]
        version = (bits[1] if len(bits) > 1 and bits[1] else DEFAULT_PROMPT_VERSION)
        # Effort is PER ARM, not per run. Three separate runs at three efforts
        # would put provider variance back in exactly where the paired design
        # takes it out - p90 moved 17.6s to 36.2s on one endpoint inside an
        # hour, which is larger than any effort effect we are looking for.
        # "none" means send no `reasoning` field at all, which is the shipping
        # default and therefore the control.
        effort = bits[2].lower() if len(bits) > 2 and bits[2] else None
        if effort in ("none", "default", ""):
            effort = None
        if effort not in (None, "low", "medium", "high"):
            sys.exit(f"arm {part!r}: unknown effort {effort!r}")
        # 4th field: the image overlay, per arm, so a grid arm and its control
        # can run in the same burst instead of hours apart.
        overlay = bits[3].lower() if len(bits) > 3 and bits[3] else "plain"
        if overlay not in ("plain", "grid", "ruler"):
            sys.exit(f"arm {part!r}: unknown overlay {overlay!r}; "
                     f"use plain, grid or ruler")
        if version not in PROMPT_SETS:
            sys.exit(f"arm {part!r}: unknown prompt version {version!r}. "
                     f"Known: {sorted(PROMPT_SETS)}")
        # A prompt set that dictates its own coordinate space overrides the
        # per-model pin: we are the ones asking for 0-1000 here, so the model's
        # natural convention is not what governs. Still verified, not assumed -
        # validate_boxes rejects anything off the declared scale.
        convention = PROMPT_SETS[version]["coords"] or COORD_CONVENTION.get(model)
        if convention is None:
            sys.exit(
                f"\narm {part!r}: {model} has no pinned coordinate convention.\n\n"
                f"An arm run cannot probe on the fly — the whole point is that\n"
                f"every arm is comparable, and a model whose convention we are\n"
                f"guessing is not comparable to one we measured. Probe it first:\n\n"
                f"    uv run detect.py {args.clip} --model {model} "
                f"--probe-convention\n\n"
                f"That costs three calls and prints the line to add to\n"
                f"COORD_CONVENTION in detect.py.\n")
        arms.append({
            "name": chr(ord("A") + n),
            "model": model,
            "version": version,
            "effort": effort,
            "overlay": overlay,
            "convention": convention,
            "schema": build_schema(scene_last=args.scene_last,
                                   terse=args.terse_schema,
                                   compact=args.compact, version=version),
        })
    if len(arms) < 2:
        sys.exit("--arms needs at least two arms; use --model for a single run.")
    return arms


def run_arms(args, arms, frames, session, headers, variant):
    """Every arm, on the same frames, in one concurrent burst.

    WHY THIS EXISTS. Provider latency is the dominant noise term — p90 went
    17.6s to 36.2s on the same endpoint one hour apart — so running arm A as a
    block and arm B as a block makes time-of-day a hidden variable perfectly
    correlated with the arm. That is how D25 came to publish a confounded table.
    Every arm sees the SAME frames here, which makes it a paired design: each
    frame is its own control.

    Two separate things are being balanced, and they need different mechanisms:

      TIME.     Handled by firing all arms in one pool. 150 calls go out inside
                a few milliseconds of each other, so all three arms meet
                identical provider conditions by construction — better than
                interleaving in sequence, not merely as good.
      POSITION. Handled by rotating the arm order per frame. Batch position is
                a measured confound in its own right (§7: frames 0-14 p50 26.1s,
                30-74 p50 10.9-13.2s, 120-149 p50 25.3s), so if arm A were
                always submitted first it would collect every slow head-of-batch
                slot. Rotating gives each arm an even spread of positions.

    Size is deliberate too: 3 arms x 50 frames = 150 concurrent 1080p uploads,
    which is the size flex_30s already ran cleanly. 300 was measured to produce
    a 25% transport-failure rate.
    """
    tasks = []
    for i, (frame_idx, path) in enumerate(frames):
        for j in range(len(arms)):
            tasks.append((arms[(i + j) % len(arms)], frame_idx, path))

    workers = args.max_concurrent or max(1, len(tasks))
    print(f"  {len(arms)} arms x {len(frames)} frames = {len(tasks)} calls, "
          f"{workers} concurrent")
    for a in arms:
        print(f"    {a['name']}: {a['model']}  prompt {a['version']}  "
              f"effort {a['effort'] or 'default'}  overlay {a['overlay']}  "
              f"({a['convention']})")

    t0 = time.perf_counter()
    records = run_pool([
        (lambda t=t: dict(call_with_retry(
            session, headers, t[0]["model"], t[1], t[2], args.width,
            args.timeout, t[0]["schema"], t[0]["effort"], t[0]["convention"],
            {**variant, "grid": t[0]["overlay"] == "grid",
             "ruler": t[0]["overlay"] == "ruler"},
            version=t[0]["version"]),
            arm=t[0]["name"], effort=t[0]["effort"],
            overlay=t[0]["overlay"]))
        for t in tasks], workers)
    wall = time.perf_counter() - t0
    return records, wall


def write_arm_outputs(args, arms, records, wall, variant, stamp):
    """One detections file per arm, each independently trackable.

    Named `<clip>__<tag>_<arm><version>.json` so the arm is visible in every
    downstream filename — a render is otherwise indistinguishable from any
    other and the comparison is lost the moment two of them are on screen.
    """
    DETECTIONS.mkdir(parents=True, exist_ok=True)
    n_source = source_frame_count(args.clip)
    written = []
    for a in arms:
        mine = [r for r in records if r.get("arm") == a["name"]]
        ok = [r for r in mine if r.get("ok")]
        dropped = [r for r in mine if not r.get("ok")]
        tag = (f"{args.tag}_{a['name']}{a['version']}_"
               f"{a['effort'] or 'def'}_{a['overlay']}")
        out = DETECTIONS / f"{args.clip.stem}__{tag}.json"
        out.write_text(json.dumps({
            "clip": args.clip.name, "model": a["model"], "fps": args.fps,
            "width": args.width, "tag": tag, "ts": stamp,
            "effort": a["effort"], "scene_last": args.scene_last,
            "terse_schema": args.terse_schema, "timeout_s": args.timeout,
            "variant": {**variant, "grid": a["overlay"] == "grid",
                        "ruler": a["overlay"] == "ruler"},
            "coord_convention": a["convention"],
            # The arm block is what makes this file self-describing. A
            # detections file that does not say which prompt produced it is
            # unusable in a comparison three days later.
            "arm": a["name"], "prompt_version": a["version"],
            "arm_set": [f"{x['model']}:{x['version']}" for x in arms],
            "source_fps": SOURCE_FPS, "wall_s": wall,
            "n_source_frames": n_source,
            "frames": [{"frame": r["frame"], **r["result"]} for r in ok],
            "dropped": [{"frame": r["frame"], "error": r.get("error")}
                        for r in dropped],
        }, indent=1), encoding="utf-8")
        written.append((a, out, ok, dropped))
    return written


def report_arms(arms, records, wall, written):
    """The comparison table. Reasoning and content tokens are reported
    SEPARATELY, which is the lesson D25 paid for: reasoning is 77.9% of output
    and ~64% of the bill, and every optimisation so far aimed at the other 18%.
    A single 'output tokens' column hides the entire effect being measured."""
    print(f"\n  wall {wall:.1f}s for all arms\n")
    hdr = (f"  {'arm':4}{'model':26}{'pv':4}{'ok':>6}{'$/call':>9}{'$/150':>8}"
           f"{'in':>7}{'reason':>8}{'content':>8}{'lat p50':>9}{'lat p90':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for a, out, ok, dropped in written:
        mine = [r for r in records if r.get("arm") == a["name"]]
        costed = [r for r in mine if r.get("cost_usd")]
        lat = sorted(r["latency_s"] for r in mine if r.get("latency_s"))
        n = len(costed) or 1
        cost = sum(r["cost_usd"] for r in costed) / n
        rea = sum(r.get("reasoning_tokens") or 0 for r in ok) / max(len(ok), 1)
        comp = sum(r.get("completion_tokens") or 0 for r in ok) / max(len(ok), 1)
        pin = sum(r.get("prompt_tokens") or 0 for r in ok) / max(len(ok), 1)
        p50 = lat[len(lat) // 2] if lat else 0
        p90 = lat[int(len(lat) * 0.9)] if lat else 0
        print(f"  {a['name']:4}{a['model'][:25]:26}{a['version']:4}"
              f"{len(ok):>3}/{len(mine):<2}{cost:9.5f}{cost*150:8.3f}"
              f"{pin:7.0f}{rea:8.0f}{comp-rea:8.0f}{p50:9.1f}{p90:9.1f}")

    print(f"\n  {'arm':4}{'players/frame':>15}{'num read':>10}{'ball':>8}"
          f"{'invalid':>9}{'providers':>28}")
    for a, out, ok, dropped in written:
        pf = [len(r["result"]["players"]) for r in ok]
        nums = sum(len([p for p in r["result"]["players"]
                        if p.get("num") is not None]) for r in ok)
        tot = sum(pf) or 1
        ball = sum(1 for r in ok if r["result"].get("ball"))
        inval = sum((r.get("invalid_boxes") or {}).get("dropped", 0) for r in ok)
        provs = {}
        for r in ok:
            provs[r.get("provider") or "?"] = provs.get(r.get("provider") or "?", 0) + 1
        ps = ", ".join(f"{k} {v}" for k, v in sorted(provs.items(),
                                                     key=lambda kv: -kv[1])[:2])
        med = sorted(pf)[len(pf) // 2] if pf else 0
        print(f"  {a['name']:4}{med:>15}{100*nums/tot:9.1f}%"
              f"{ball:>5}/{len(ok):<3}{inval:>8}{ps:>28}")
    for a, out, ok, dropped in written:
        print(f"\n  arm {a['name']}: {out}")
        if dropped:
            why = {}
            for r in dropped:
                k = (r.get("error") or "?").split(":")[0]
                why[k] = why.get(k, 0) + 1
            print(f"    dropped {len(dropped)}: {why}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--fps", type=int, default=10,
                    help="frames sampled per second (default 10)")
    ap.add_argument("--width", type=int, default=None,
                    help="downscale width; omit for native (recommended)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--tag", default="dev", help="run tag, for the ablation log")
    ap.add_argument("--frames", default=None,
                    help="comma-separated source indices — smoke test on a few")
    # --- ablation switches (docs/decisions.md, A1-A7) ---------------------
    ap.add_argument("--effort", choices=["low", "medium", "high"], default=None,
                    help="A5: reasoning effort. Reasoning is 64%% of output "
                         "tokens on Luna, so this is the largest latency lever.")
    ap.add_argument("--scene-last", action="store_true",
                    help="A4: move the `scene` scratchpad to the END of the "
                         "schema, so coordinates are produced with no thinking "
                         "tokens spent first.")
    ap.add_argument("--terse-schema", action="store_true",
                    help="A7: strip the long field descriptions. Prompt+schema "
                         "costs ~2100 input tokens per call.")
    ap.add_argument("--max-concurrent", type=int, default=None,
                    help="cap simultaneous calls. Unset = all at once, which is "
                         "fine at 720p and caused a 25%% connection-failure rate "
                         "at 1080p. Try 64 for large frames.")
    # --- the four container/format ablations. All default OFF. -----------
    ap.add_argument("--container", action="store_true",
                    help="field rules live only in the schema, not the prompt")
    ap.add_argument("--system", action="store_true",
                    help="instructions in a system message, image alone in user")
    ap.add_argument("--image-first", action="store_true",
                    help="image before text, so instructions are most recent")
    ap.add_argument("--compact", action="store_true",
                    help="players as fixed-order arrays, not named objects")
    ap.add_argument("--ruler", action="store_true",
                    help="A1: draw a coordinate ruler on the frame")
    ap.add_argument("--grid", action="store_true",
                    help="A1b: overlay a thin magenta grid every 0.1")
    ap.add_argument("--probe-convention", action="store_true",
                    help="send 3 frames, report which coordinate convention "
                         "this model uses, and exit without pinning anything")
    # Routing was an unnoticed variable for the whole project. Gemini serves the
    # SAME model at three prices (flex 0.38/1.88, standard 0.75/3.75, priority
    # 1.35/6.75) and default routing silently moved from flex to standard between
    # 28 and 31 Aug: 511 calls billed at 1.88, then 562 at 3.75, same model, same
    # clip. That looked like a price rise and was recorded as one. It was routing.
    # Both Google flex endpoints, in preference order. They are priced
    # IDENTICALLY (0.375/1.875), so listing the second costs nothing and gives a
    # rate-limited call somewhere to go that is not the 2x standard tier.
    # allow_fallbacks stays false, so standard remains unreachable by accident.
    ap.add_argument("--provider-order",
                    default="google-ai-studio/flex,google-vertex/global/flex",
                    help="comma-separated OpenRouter provider tags to pin. "
                         "Defaults to both Google FLEX endpoints (same price). "
                         "Sets allow_fallbacks=false so a run cannot silently "
                         "land on a dearer tier. Pass '' to disable pinning")
    ap.add_argument("--cut-share", type=float, default=None,
                    help="fraction of calls that must return before the "
                         "stragglers are abandoned. Pass 1.0 to DISABLE the "
                         "cut and record the full latency distribution, which "
                         "is what makes the saving measurable offline "
                         "afterwards - an abandoned call's true latency is "
                         "unknowable.")
    ap.add_argument("--prompt-version", choices=sorted(PROMPT_SETS),
                    default=DEFAULT_PROMPT_VERSION,
                    help="v2 (default) is the D26 rewrite. v1 is the frozen "
                         "pre-D26 prompt and schema from prompt_v1.py, which "
                         "produced every measurement before 3 Sep.")
    ap.add_argument("--arms", default=None,
                    help="Run several model/prompt combinations on the SAME "
                         "frames in one concurrent burst, e.g. "
                         "'google/gemini-3.7-flash:v2,google/gemini-3.8-flash:v1"
                         ",google/gemini-3.8-flash:v2'. Writes one detections "
                         "file per arm. Overrides --model and --prompt-version.")
    args = ap.parse_args()

    if args.cut_share is not None:
        globals()["CUT_SHARE"] = args.cut_share
    if args.arms and args.probe_convention:
        sys.exit("--arms and --probe-convention are mutually exclusive: probe "
                 "each model on its own, pin it, then run the arms.")

    schema = build_schema(scene_last=args.scene_last, terse=args.terse_schema,
                          compact=args.compact, version=args.prompt_version)
    variant = {"container": args.container, "system": args.system,
               "image_first": args.image_first, "compact": args.compact,
               "ruler": args.ruler, "grid": args.grid,
               "provider_order": ([p.strip() for p in args.provider_order.split(",")]
                                  if args.provider_order else None)}

    arms = parse_arms(args.arms, args) if args.arms else None

    # Pinned or it does not run. No inference, no fallback, no "probably".
    # parse_arms has already enforced this per arm.
    convention = (PROMPT_SETS[args.prompt_version]["coords"]
                  or COORD_CONVENTION.get(args.model))
    if convention is None and not args.probe_convention and not arms:
        sys.exit(
            f"\n{args.model} has no pinned coordinate convention.\n\n"
            f"Models do not reliably obey the fraction convention the schema\n"
            f"asks for, and guessing per-response is how the last project lost\n"
            f"days. Find out once, then pin it:\n\n"
            f"    uv run detect.py {args.clip} --model {args.model} "
            f"--probe-convention\n\n"
            f"That costs three calls and prints the line to add to\n"
            f"COORD_CONVENTION in detect.py.\n")

    if not args.clip.exists():
        sys.exit(f"no such clip: {args.clip}")

    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("No API key in the environment. Nothing was read or printed.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    del key

    frames_dir = Path("outputs/frames") / args.clip.stem
    print(f"extracting {args.fps}fps from {args.clip.name} ...")
    _t_ex0 = time.perf_counter()
    frames = extract_frames(args.clip, args.fps, frames_dir)
    T_EXTRACT[0] = time.perf_counter() - _t_ex0
    if args.frames:
        want = {int(x) for x in args.frames.split(",")}
        frames = [(i, p) for i, p in frames if i in want]
    if args.probe_convention:
        step = max(1, len(frames) // 3)
        frames = frames[::step][:3]
        print("  PROBE: 3 frames, raw coordinates, nothing will be converted")
    print(f"  {len(frames)} frames to send"
          f"{' (native resolution)' if not args.width else f' at {args.width}px'}")

    n_conn = (len(frames) * len(arms)) if arms else len(frames)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=n_conn or 1, pool_maxsize=n_conn or 1))

    if arms:
        stamp = datetime.now(timezone.utc).isoformat()
        records, wall = run_arms(args, arms, frames, session, headers, variant)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({
                    "stage": "detect", "tag": args.tag, "ts": stamp,
                    "clip": args.clip.name, "fps": args.fps,
                    "width": args.width,
                    **{k: v for k, v in r.items() if k != "result"}}) + "\n")
        written = write_arm_outputs(args, arms, records, wall, variant, stamp)
        report_arms(arms, records, wall, written)
        return

    # Concurrency is free on latency — measured: median call time was flat to
    # N=64 and wall clock is set by the slowest call, not the queue. But it is
    # NOT free on the network: 300 simultaneous 1080p uploads is ~120MB leaving
    # at once and it produced a 25% connection-failure rate. Cap it when the
    # payload is large.
    workers = args.max_concurrent or max(1, len(frames))
    t0 = time.perf_counter()
    records = run_pool([
        (lambda fp=fp: call_with_retry(session, headers, args.model, fp[0], fp[1],
                                       args.width, args.timeout, schema,
                                       args.effort, convention, variant,
                                       version=args.prompt_version))
        for fp in frames], workers)
    wall = time.perf_counter() - t0

    ok = [r for r in records if r.get("ok")]
    dropped = [r for r in records if not r.get("ok")]
    stamp = datetime.now(timezone.utc).isoformat()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"stage": "detect", "tag": args.tag, "ts": stamp,
                                "clip": args.clip.name, "model": args.model,
                                "fps": args.fps, "width": args.width,
                                **{k: v for k, v in r.items() if k != "result"}}) + "\n")

    # MOVED BELOW THE LOG WRITE, 4 Sep. The probe used to return here, three
    # lines earlier, which had two consequences both already written down as
    # defects and neither connected to this `return`:
    #
    #   - HANDOFF §8: a FAILED probe reported only "no usable detections" and
    #     discarded the per-call errors, which is why kimi-k2.5 and glm-5.3-flash
    #     were dropped undiagnosed after two attempts each.
    #   - D21: the budget ledger was incomplete because some writers never
    #     recorded cost. Every probe ever run is one of those writers. They are
    #     cheap - 3 calls - but the ledger's value is being complete, not being
    #     approximately right, and "probes are small" is exactly the reasoning
    #     that left probe_budget.py uncosted.
    if args.probe_convention:
        report_convention(records, args.model)
        for r in dropped:
            print(f"    frame {r.get('frame')}: HTTP {r.get('status')} "
                  f"{(r.get('error') or '')[:120]}")
        spent = sum(r.get("cost_usd") or 0 for r in records)
        print(f"\n  logged {len(records)} probe calls to {LOG}, ${spent:.4f}")
        return

    DETECTIONS.mkdir(parents=True, exist_ok=True)
    out = DETECTIONS / f"{args.clip.stem}__{args.tag}.json"
    _t_w0 = time.perf_counter()
    out.write_text(json.dumps({
        "clip": args.clip.name, "model": args.model, "fps": args.fps,
        "width": args.width, "tag": args.tag, "ts": stamp,
        "effort": args.effort, "scene_last": args.scene_last,
        "terse_schema": args.terse_schema, "timeout_s": args.timeout,
        "variant": variant,
        # Which prompt+schema package produced this. Every detections file
        # written before 4 Sep lacks the key and is v1 by definition.
        "prompt_version": args.prompt_version,
        # Recorded so a result can never be misread later, and so a convention
        # change shows up as a diff rather than as mysteriously bad tracking.
        "coord_convention": convention,
        "source_fps": SOURCE_FPS, "wall_s": wall,
        "n_source_frames": source_frame_count(args.clip),
        "frames": [{"frame": r["frame"], **r["result"]} for r in ok],
        "dropped": [{"frame": r["frame"], "error": r.get("error")} for r in dropped],
    }, indent=1), encoding="utf-8")
    T_WRITE[0] = time.perf_counter() - _t_w0

    lat = sorted(r["latency_s"] for r in records if r.get("latency_s"))
    print(f"\n  wall            {wall:.1f}s")
    print(f"  returned        {len(ok)}/{len(records)}")
    if dropped:
        why = {}
        for r in dropped:
            why[r.get("error", "?").split(":")[0]] = why.get(
                r.get("error", "?").split(":")[0], 0) + 1
        print(f"  dropped         {len(dropped)}  {why}")
    if lat:
        print(f"  latency med     {lat[len(lat)//2]:.1f}s   max {lat[-1]:.1f}s")
    retried = sum(1 for r in records if r.get("retries"))
    if retried:
        print(f"  retried         {retried} calls retried")
    # Backoff is reported SEPARATELY from latency and named against wall clock,
    # so "did the retries cause this?" is answerable from the run summary alone.
    # latency_s excludes the sleeps by construction (t0 is set after them).
    waits = [r["retry_wait_s"] for r in records if r.get("retry_wait_s")]
    if waits:
        print(f"  429 backoff     {len(waits)} calls slept, "
              f"{sum(waits):.1f}s total, worst {max(waits):.1f}s on one call")
        print(f"                  wall was {wall:.1f}s; without any backoff the "
              f"floor would be ~{wall - max(waits):.1f}s")
    cutoff = [r for r in records if (r.get("error") or "") == "straggler cut"]
    if cutoff:
        kept = sorted(r["latency_s"] for r in records
                      if r.get("ok") and r.get("latency_s"))
        print(f"  straggler cut   {len(cutoff)} calls abandoned at "
              f"{CUT_SHARE:.0%} + {CUT_GRACE_S}s; slowest KEPT call "
              f"{kept[-1]:.1f}s. Their true latency is unknowable, so the "
              f"saving cannot be measured from this run")
    mal = [r for r in records if r.get("malformed_count")]
    if mal:
        print(f"  malformed rows  {sum(r['malformed_count'] for r in mal)} player "
              f"rows of the wrong shape across {len(mal)} frames — the model "
              f"changed output format mid-run")
        for r in mal[:3]:
            print(f"                  frame {r['frame']}: {r['malformed_rows']}")
    inval = [r for r in records if r.get("invalid_boxes")]
    if inval:
        nb = sum(r["invalid_boxes"]["dropped"] for r in inval)
        nw = sum(r["invalid_boxes"].get("wide", 0) for r in inval)
        nf = sum(1 for r in inval if r["invalid_boxes"].get("frame_rejected"))
        print(f"  invalid boxes   {nb} discarded across {len(inval)} frames"
              f"  ({nb - nw} off-scale, {nw} over {MAX_PIXEL_ASPECT}:1;"
              f" {nf} frames rejected outright)")
    if ok:
        players = [len(r["result"]["players"]) for r in ok]
        with_ball = sum(1 for r in ok if r["result"]["ball"])
        with_num = sum(len([p for p in r["result"]["players"] if p["num"] is not None])
                       for r in ok)
        kits = {}
        for r in ok:
            for p in r["result"]["players"]:
                kits[p["kit"]] = kits.get(p["kit"], 0) + 1
        print(f"  players/frame   min {min(players)} med "
              f"{sorted(players)[len(players)//2]} max {max(players)}")
        print(f"  ball seen       {with_ball}/{len(ok)} frames")
        print(f"  numbers read    {with_num} of {sum(players)} player sightings")
        print(f"  kit colours     {dict(sorted(kits.items(), key=lambda k: -k[1]))}")
        # Box sanity. A player in a wide shot is a tall thin sliver: h roughly
        # 0.08-0.35, w roughly 0.02-0.08. If h comes back ~0.9, or w > h, the
        # model is returning corners (x2,y2) instead of extents (w,h) and every
        # downstream foot position is wrong.
        hs = sorted(p["h"] for r in ok for p in r["result"]["players"])
        ws = sorted(p["w"] for r in ok for p in r["result"]["players"])
        if hs:
            print(f"  box h           med {hs[len(hs)//2]:.3f}  "
                  f"(expect ~0.08-0.35 in a wide shot)")
            print(f"  box w           med {ws[len(ws)//2]:.3f}  "
                  f"(expect < box h; taller than wide)")
        tok = [r.get("completion_tokens") or 0 for r in ok]
        print(f"  output tokens   med {sorted(tok)[len(tok)//2]}")
        gk = sum(1 for r in ok for p in r["result"]["players"]
                 if p.get("role") == "goalkeeper")
        print(f"  goalkeepers     {gk} sightings across {len(ok)} frames")
        costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
        if costs:
            print(f"  COST           ${sum(costs):.4f} this run  "
                  f"(${sum(costs)/max(len(costs),1)*1000:.2f} per 1000 frames)")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
