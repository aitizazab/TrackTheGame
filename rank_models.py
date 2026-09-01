"""Rank the vision catalogue by what a finished video would actually cost.

$/Mtok is not a cost model. What we spend is set by how many tokens ONE FRAME
costs and how many frames a video needs, so the catalogue price has to be turned
into dollars-per-video before any two models can be compared.

Measured on the luna_v2 run, 300 frames at native 1280x720:

    input   2188 tok/frame  ->  656k per video   (image ~1120 + prompt/schema ~1070)
    output  2035 tok/frame  ->  610k per video   (of which 1313 was REASONING)

    $/video = 0.656 * in_$Mtok + 0.610 * out_$Mtok

That predicts $0.86 for Luna against $0.955 actually billed — close enough to
rank with, and the gap is cache and provider overhead.

Two scenarios are reported, because they rank differently:

    reasoning   2035 output tokens/frame, as Luna behaves
    terse       ~750 output tokens/frame, what a non-reasoning model emits for
                the same JSON. Models without reasoning are far cheaper than
                their headline price suggests, and models with it far worse.

Hard requirements applied as filters, not preferences:
  - accepts image input
  - json_schema structured output (otherwise it is prompt-and-parse with
    retries, a different mechanism that would confound the comparison)
  - max output tokens >= 8000 (Luna needed 6500; anything under truncates)

    uv run rank_models.py
    uv run rank_models.py --budget 1.00 --clip-seconds 10
"""

import argparse
import json
from pathlib import Path

SRC = Path("docs/vision_models.json")
IN_TOK, OUT_TOK_REASON, OUT_TOK_TERSE = 2188, 2035, 750
MIN_MAX_OUTPUT = 8000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--clip-seconds", type=int, default=30)
    ap.add_argument("--budget", type=float, default=1.00)
    ap.add_argument("--all", action="store_true", help="include no-json models")
    args = ap.parse_args()

    frames = args.fps * args.clip_seconds
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for m in data["models"]:
        if m["in_per_m"] < 0:          # openrouter/auto sentinel rows
            continue
        if not args.all and not m["structured"]:
            continue
        if (m["max_output"] or 0) < MIN_MAX_OUTPUT:
            continue
        inp = frames * IN_TOK / 1e6
        rsn = inp * m["in_per_m"] + frames * OUT_TOK_REASON / 1e6 * m["out_per_m"]
        trs = inp * m["in_per_m"] + frames * OUT_TOK_TERSE / 1e6 * m["out_per_m"]
        rows.append({**m, "cost_reason": rsn, "cost_terse": trs})

    rows.sort(key=lambda r: r["cost_terse"])
    fit = [r for r in rows if r["cost_terse"] <= args.budget]

    print(f"{args.clip_seconds}s clip at {args.fps}fps = {frames} frames")
    print(f"filters: image input + json_schema + max_output >= {MIN_MAX_OUTPUT}")
    print(f"{len(rows)} models qualify; {len(fit)} fit ${args.budget:.2f}/video terse\n")
    print(f"{'model':<46} {'$/vid terse':>11} {'$/vid rsn':>10} {'maxout':>7} {'rsn':>4}")
    print("-" * 84)
    for r in rows:
        flag = "" if r["cost_terse"] <= args.budget else "  over"
        print(f"{r['id'][:46]:<46} {r['cost_terse']:>11.3f} {r['cost_reason']:>10.3f} "
              f"{str(r['max_output']):>7} {'yes' if r['reasoning'] else '-':>4}{flag}")


if __name__ == "__main__":
    main()
