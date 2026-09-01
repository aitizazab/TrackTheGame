"""Enumerate every OpenRouter model that accepts image input.

The catalogue endpoint is public — no key, no auth, no spend. That matters: this
is the one piece of model research that costs nothing, so it should be
exhaustive rather than a shortlist someone half-remembered.

For each model it records what actually decides whether it can do this task:

  input modalities   does it take images at all
  pricing            per input token, per output token, and per image where the
                     provider charges images separately
  context length     an image at 1280x720 costs ~1120 tokens on top of the prompt
  max output tokens  Luna needed 6500; a model capped at 4096 cannot answer
  structured output  json_schema support. Without it we are on prompt-and-parse
                     with retries, which is a different mechanism and confounds
                     any model comparison that mixes the two
  reasoning support  reasoning was 63% of Luna's output tokens, so being able to
                     turn it down is a first-class cost and latency lever

    uv run list_vision_models.py                 # summary + write JSON
    uv run list_vision_models.py --all           # print every row
    uv run list_vision_models.py --max-cost 2.0  # under $2 per Mtok input
"""

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CATALOGUE = "https://" + "openrouter" + ".ai/api/v1/models"
OUT = Path("docs/vision_models.json")


def fetch():
    req = urllib.request.Request(CATALOGUE, headers={"User-Agent": "trackthegame/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))["data"]


def money(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def row(m):
    arch = m.get("architecture") or {}
    pricing = m.get("pricing") or {}
    top = m.get("top_provider") or {}
    params = set(m.get("supported_parameters") or [])
    mods = arch.get("input_modalities") or []
    return {
        "id": m.get("id"),
        "name": m.get("name"),
        "created": m.get("created"),
        "modalities": mods,
        "output_modalities": arch.get("output_modalities") or [],
        "context": m.get("context_length") or top.get("context_length"),
        "max_output": top.get("max_completion_tokens"),
        # per-token prices are per single token; x1e6 gives the usual $/Mtok
        "in_per_m": (money(pricing.get("prompt")) or 0) * 1e6,
        "out_per_m": (money(pricing.get("completion")) or 0) * 1e6,
        "per_image": money(pricing.get("image")),
        "structured": "structured_outputs" in params,
        "response_format": "response_format" in params,
        "reasoning": "reasoning" in params or "include_reasoning" in params,
        "tools": "tools" in params,
        "temperature": "temperature" in params,
        "moderated": bool(top.get("is_moderated")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="only models at or under this $/Mtok input")
    args = ap.parse_args()

    models = [row(m) for m in fetch()]
    vision = [r for r in models
              if any(x in ("image", "images") for x in r["modalities"])]
    vision.sort(key=lambda r: (r["in_per_m"], r["out_per_m"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"fetched": datetime.now(timezone.utc).isoformat(),
         "total_models": len(models), "vision_models": len(vision),
         "models": vision}, indent=1), encoding="utf-8")

    free = [r for r in vision if r["in_per_m"] == 0]
    print(f"catalogue: {len(models)} models, {len(vision)} accept image input "
          f"({len(free)} at zero listed cost)")
    print(f"  structured outputs: {sum(1 for r in vision if r['structured'])}")
    print(f"  reasoning control:  {sum(1 for r in vision if r['reasoning'])}")
    print(f"  -> {OUT}")

    shown = vision if args.all else vision[:400]
    if args.max_cost is not None:
        shown = [r for r in shown if r["in_per_m"] <= args.max_cost]
    print(f"\n{'model id':<52} {'$/Mtok in':>9} {'out':>8} {'ctx':>8} "
          f"{'maxout':>7} {'json':>5} {'rsn':>4}")
    print("-" * 100)
    for r in shown:
        print(f"{(r['id'] or '')[:52]:<52} {r['in_per_m']:>9.3f} "
              f"{r['out_per_m']:>8.3f} {str(r['context'] or '-'):>8} "
              f"{str(r['max_output'] or '-'):>7} "
              f"{'yes' if r['structured'] else '-':>5} "
              f"{'yes' if r['reasoning'] else '-':>4}")


if __name__ == "__main__":
    main()
