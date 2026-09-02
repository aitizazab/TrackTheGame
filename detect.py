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

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
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
MAX_TOKENS = 6500

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
        "required": ["scene", "kits", "players", "ball"],
        "properties": {
            "scene": {
                "type": "string",
                "description": "One sentence: camera framing, lighting, and the "
                               "two kit colours. Written BEFORE looking for "
                               "positions, as working-out."
            },
            "kits": {
                "type": "array",
                "description": "The distinct outfield kits visible, most common "
                               "first. Usually exactly two.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["colour", "accent"],
                    "properties": {
                        "colour": {"type": "string",
                                   "description": "dominant shirt colour, one "
                                                  "common word"},
                        # The renderer needs a second colour to fall back on when
                        # the two kits are too close to tell apart at a glance.
                        # Asked once per frame rather than once per player: it is
                        # a property of the kit, and per-player would cost ~20
                        # extra output tokens per person for no extra signal.
                        "accent": {"type": ["string", "null"],
                                   "description": "secondary colour on that kit "
                                                  "— trim, sleeves, shorts, or "
                                                  "the number itself. null if "
                                                  "the kit is plain."}
                    }
                }
            },
            "players": {
                "type": "array",
                "description": "One entry per player on the field of play, "
                               "GOALKEEPERS INCLUDED. Empty array is valid and "
                               "correct if none are visible.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x", "y", "w", "h", "kit", "num", "role", "conf"],
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
                                "description": "Shirt colour as one common word"},
                        "num": {"type": ["integer", "null"],
                                "description": "Jersey number ONLY if you can "
                                               "actually read it. null otherwise."},
                        # Without this a goalkeeper in a third kit colour is
                        # indistinguishable from an unstable colour word, and the
                        # renderer cannot tell "green kit = keeper" from "someone
                        # said green once by mistake".
                        "role": {"type": "string", "enum": ["outfield", "goalkeeper"],
                                 "description": "goalkeeper if they wear a "
                                                "different kit from both teams "
                                                "and stand in/near a goal. In "
                                                "sports with no goalkeeper, "
                                                "every player is outfield"},
                        "conf": {"type": "number",
                                 "description": "0.0 to 1.0, how sure you are this "
                                                "is a player at this position"}
                    }
                }
            },
            "ball": {
                "type": ["object", "null"],
                "description": "null when the ball is not visible. It often is not.",
                "additionalProperties": False,
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

# Every clause here is defending against a specific failure we predicted:
# hallucinated players (schemas compel an answer), inconsistent team naming
# across independent calls, guessed jersey numbers, and phantom balls.
PROMPT = """You are looking at one frame of sports footage.

Report the PLAYERS and the BALL.

Give each one a BOUNDING BOX in fractions of the image, never in pixels.
  x = left edge of the box    (0.0 = image left,  1.0 = image right)
  y = top edge of the box     (0.0 = image top,   1.0 = image bottom)
  w = box width               (as a fraction of the image width)
  h = box height              (as a fraction of the image height)

  Worked example. A player standing in the middle of the picture, occupying the
  lower half vertically and a narrow slice horizontally:
      x = 0.48, y = 0.50, w = 0.04, h = 0.28
  Their box therefore spans 0.48-0.52 across and 0.50-0.78 down.

  The box must be TIGHT: top edge at the top of their head, bottom edge where
  their feet meet the ground. The bottom edge is used to place a marker under
  them, so if the box runs long the marker floats below their feet.

For each player also give:
  kit  the colour of their SHIRT, as one ordinary word: red, blue, white,
       yellow, green, black, orange, purple. Judge the colour itself. Do not
       call them "team A" or "home"; another frame will be judged separately
       and the colours must agree between them.
  num  the number on their shirt ONLY IF YOU CAN GENUINELY READ IT. If their
       back is turned, if it is blurred, if they are too small, if it is
       covered - use null. A wrong number is far worse than no number.
  conf 1.0 you are certain, 0.5 you think so, 0.2 you are guessing.

Rules that matter:
  - Report players on the field of play - the pitch, court, or playing surface.
  - GOALKEEPERS. If this sport has a goalkeeper (football, hockey, handball,
    futsal), they ARE players and must be reported even though their kit matches
    neither team. Mark them role="goalkeeper". If the sport has no goalkeeper
    (basketball, volleyball), every player is role="outfield". Decide from what
    you can see in the image; do not assume the sport.
  - Do NOT report match officials (referees, umpires, linesmen), substitutes or
    players on the bench, coaches, medical staff, the crowd, or ball boys.
  - Report every player you can see, including partly hidden ones. Give a partly
    hidden player a low conf rather than leaving them out.
  - Do NOT pad the list to a round number. If you can see 7 players, report 7.
    There is no expected count, and it does not depend on the sport.

THE BALL:
  - A tight box, same fraction format.
  - Markings painted on the playing surface are not the ball - centre spots,
    penalty spots, painted arcs, court lines and logos. Check that what you are
    looking at sits ABOVE the surface rather than being printed onto it.
  - If you cannot see the ball, set ball to null. Do not place it where you
    think it ought to be, and do not settle for the nearest small round thing.

In "kits", list the two teams' kits. For each, give the dominant shirt colour and
one secondary "accent" colour — the trim, sleeves, shorts, or the colour the
numbers are printed in. If a kit is genuinely plain, accent is null.

Fill in "scene" first, as working-out, before you give any coordinates."""


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


RULER_NOTE = """
A COORDINATE RULER is drawn on the image. Yellow numerals along the top edge
mark x = .1 to .9; along the left edge they mark y = .1 to .9. Read positions
off it rather than estimating them. The ruler itself is not part of the scene —
do not report it as a player or as the ball."""


def encode(path: Path, width: int = None, ruler: bool = False) -> tuple:
    img = Image.open(path).convert("RGB")
    if width and width != img.width:
        img = img.resize((width, round(width * img.height / img.width)))
    if ruler:
        img = draw_ruler(img)
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


def validate_boxes(result: dict) -> dict:
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
    """
    players = result.get("players") or []
    total = len(players)
    ok = []
    for p in players:
        vals = (p.get("x"), p.get("y"),
                (p.get("x") or 0) + (p.get("w") or 0),
                (p.get("y") or 0) + (p.get("h") or 0))
        if all(v is not None and COORD_MIN <= v <= COORD_MAX for v in vals) \
                and (p.get("w") or 0) > 0 and (p.get("h") or 0) > 0:
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
    return {"total": total, "dropped": dropped, "ball_dropped": ball_dropped,
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
COMPACT_ORDER = ["x", "y", "w", "h", "kit", "num", "role", "conf"]
COMPACT_SCHEMA = {
    "name": "frame_detections_compact",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene", "kits", "players", "ball"],
        "properties": {
            "scene": {"type": "string",
                      "description": "One sentence of working-out, written first"},
            "kits": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["colour", "accent"],
                "properties": {"colour": {"type": "string"},
                               "accent": {"type": ["string", "null"]}}}},
            "players": {
                "type": "array",
                "description": ("One array per player, ALWAYS in this order: "
                                "[x, y, w, h, kit, num, role, conf]. "
                                "x,y = box top-left as fractions 0-1. "
                                "w,h = box extents as fractions. "
                                "kit = shirt colour, one common word. "
                                "num = jersey number or null. "
                                "role = \"outfield\" or \"goalkeeper\". "
                                "conf = 0-1."),
                "items": {"type": "array",
                          "items": {"type": ["number", "string", "null"]}}},
            "ball": {"type": ["array", "null"],
                     "description": "[x, y, w, h, conf] as fractions, or null",
                     "items": {"type": ["number", "null"]}},
        }}}


def normalise_result(result: dict, compact: bool) -> dict:
    """Turn a compact array response back into the standard dict shape.

    Everything downstream — the validator, the tracker, the renderer — keeps
    working on one format. The wire format is an ablation; the internal one is
    not, and letting a switch leak past this function would mean testing the
    output format and the whole pipeline at the same time.
    """
    if not compact:
        return result
    out = []
    for row in result.get("players") or []:
        if not isinstance(row, list) or len(row) < len(COMPACT_ORDER):
            continue
        p = dict(zip(COMPACT_ORDER, row))
        for k in ("x", "y", "w", "h", "conf"):
            try:
                p[k] = float(p[k])
            except (TypeError, ValueError):
                p[k] = 0.0
        try:
            p["num"] = int(p["num"]) if p["num"] is not None else None
        except (TypeError, ValueError):
            p["num"] = None
        p["kit"] = str(p.get("kit") or "")
        out.append(p)
    result["players"] = out
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
                 compact: bool = False) -> dict:
    """SCHEMA with the two ablation knobs applied (A4 and A7).

    Field ORDER is semantically load-bearing, not cosmetic: JSON emits fields in
    schema order, so moving `scene` to the end means the model must commit to
    every coordinate before writing a word of working-out. That is the whole
    hypothesis of A4, which is why this rebuilds the dict rather than mutating
    a shared one.
    """
    s = copy.deepcopy(COMPACT_SCHEMA if compact else SCHEMA)
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
        order = ["kits", "players", "ball", "scene"]
        s["schema"]["properties"] = {k: props[k] for k in order}
        s["schema"]["required"] = order
        s["name"] = "frame_detections_scene_last"
    return s


def call_with_retry(session, headers, model, frame_idx, path, width, timeout,
                    schema, effort, convention, variant=None, tries=3):
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
                       schema, effort, convention, variant)
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
             schema, effort, convention, variant=None):
    v = variant or {}
    data_url, nbytes, w, h = encode(path, width, v.get("ruler", False))
    prompt = CONTAINER_PROMPT if v.get("container") else PROMPT
    if v.get("ruler"):
        prompt = prompt + "\n" + RULER_NOTE
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
    rec = {"frame": frame_idx, "w": w, "h": h, "bytes": nbytes}
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
        r = session.post(ENDPOINT, headers=headers, json=body,
                         timeout=(min(30.0, timeout), timeout), stream=True)
        rec["status"] = r.status_code
        if r.status_code != 200:
            rec["ok"] = False
            rec["latency_s"] = time.perf_counter() - t0
            rec["error"] = r.text[:200]
            return rec
        parts = []
        for chunk in r.iter_content(16384):
            if time.perf_counter() - t0 > timeout:
                r.close()
                rec["ok"] = False
                rec["latency_s"] = time.perf_counter() - t0
                rec["error"] = "deadline"
                return rec
            parts.append(chunk)
        rec["latency_s"] = time.perf_counter() - t0
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
        rec["result"] = normalise_result(json.loads(content), v.get("compact", False))
        if convention:
            rec["result"] = to_fractions(rec["result"], convention, w, h)
        bad = validate_boxes(rec["result"])
        if bad:
            rec["invalid_boxes"] = bad
            if bad.get("frame_rejected"):
                rec["ok"] = False
                rec["error"] = (f"coordinate corruption: {bad['dropped']} of "
                                f"{bad['total']} boxes off the 0..1 scale")
                return rec
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
    args = ap.parse_args()

    schema = build_schema(scene_last=args.scene_last, terse=args.terse_schema,
                          compact=args.compact)
    variant = {"container": args.container, "system": args.system,
               "image_first": args.image_first, "compact": args.compact,
               "ruler": args.ruler,
               "provider_order": ([p.strip() for p in args.provider_order.split(",")]
                                  if args.provider_order else None)}

    # Pinned or it does not run. No inference, no fallback, no "probably".
    convention = COORD_CONVENTION.get(args.model)
    if convention is None and not args.probe_convention:
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
    frames = extract_frames(args.clip, args.fps, frames_dir)
    if args.frames:
        want = {int(x) for x in args.frames.split(",")}
        frames = [(i, p) for i, p in frames if i in want]
    if args.probe_convention:
        step = max(1, len(frames) // 3)
        frames = frames[::step][:3]
        print("  PROBE: 3 frames, raw coordinates, nothing will be converted")
    print(f"  {len(frames)} frames to send"
          f"{' (native resolution)' if not args.width else f' at {args.width}px'}")

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=len(frames) or 1, pool_maxsize=len(frames) or 1))

    # Concurrency is free on latency — measured: median call time was flat to
    # N=64 and wall clock is set by the slowest call, not the queue. But it is
    # NOT free on the network: 300 simultaneous 1080p uploads is ~120MB leaving
    # at once and it produced a 25% connection-failure rate. Cap it when the
    # payload is large.
    workers = args.max_concurrent or max(1, len(frames))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(
            lambda fp: call_with_retry(session, headers, args.model, fp[0], fp[1],
                                       args.width, args.timeout, schema,
                                       args.effort, convention, variant),
            frames))
    wall = time.perf_counter() - t0

    if args.probe_convention:
        report_convention(records, args.model)
        return

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

    DETECTIONS.mkdir(parents=True, exist_ok=True)
    out = DETECTIONS / f"{args.clip.stem}__{args.tag}.json"
    out.write_text(json.dumps({
        "clip": args.clip.name, "model": args.model, "fps": args.fps,
        "width": args.width, "tag": args.tag, "ts": stamp,
        "effort": args.effort, "scene_last": args.scene_last,
        "terse_schema": args.terse_schema, "timeout_s": args.timeout,
        "variant": variant,
        # Recorded so a result can never be misread later, and so a convention
        # change shows up as a diff rather than as mysteriously bad tracking.
        "coord_convention": convention,
        "source_fps": SOURCE_FPS, "wall_s": wall,
        "n_source_frames": source_frame_count(args.clip),
        "frames": [{"frame": r["frame"], **r["result"]} for r in ok],
        "dropped": [{"frame": r["frame"], "error": r.get("error")} for r in dropped],
    }, indent=1), encoding="utf-8")

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
    inval = [r for r in records if r.get("invalid_boxes")]
    if inval:
        nb = sum(r["invalid_boxes"]["dropped"] for r in inval)
        nf = sum(1 for r in inval if r["invalid_boxes"].get("frame_rejected"))
        print(f"  off-scale       {nb} boxes discarded across {len(inval)} frames"
              f"  ({nf} frames rejected outright)")
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
