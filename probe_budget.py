"""
Measure the three quantities that decide the Track-the-Game architecture.

Not a speed test. A speed test tells you your line rate to some CDN, which is
not what you are spending. What you are spending is: time before the model
starts answering, tokens generated one after another inside a single call, and
whatever ceiling the provider puts on how many calls you may have open at once.

  size    Same prompt, same output cap, image sent at several resolutions.
          Plots TIME-TO-FIRST-DATA against bytes uploaded. Slope is the marginal
          cost of a byte (upload + prefill together, which is what you actually
          pay); intercept is fixed per-call overhead.

  output  Same image, max_tokens swept 1 -> 512. Slope is generation tokens/sec.
          Generation inside one request is SERIAL, so this decides whether
          packing many frames into one call is affordable.

  conc    N identical calls fired at once, N swept. Wall clock against N shows
          where the provider throttles you. Perfect parallelism is a flat line.

THREE TIMESTAMPS, because they measure different things and conflating them is
how the first version of this script lied:

  t_first_byte     any line at all, including OpenRouter's ": OPENROUTER
                   PROCESSING" keepalive comments. This is ~connection setup.
                   It is NOT upload time. Reported only to prove it is flat.
  t_first_data     first real `data:` payload. Upload + queue + prefill.
                   THIS is the one that should scale with image size.
  t_first_content  first token of actual output (content or reasoning delta).

Token usage is requested from the API, so reasoning-token spend is measured
rather than guessed at.

The key is read from the environment and never printed, logged, or returned.

    uv run probe_budget.py size
    uv run probe_budget.py output
    uv run probe_budget.py conc
    uv run probe_budget.py all --model openai/gpt-5.6-luna
    uv run probe_budget.py size --image path/to/a/real/frame.jpg
    uv run probe_budget.py size --no-reasoning
"""

import argparse
import base64
import io
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
OUT = Path("docs/budget_probe.jsonl")

SIZE_WIDTHS = [320, 512, 768, 1024, 1280]
OUTPUT_CAPS = [1, 64, 128, 256, 512]
CONC_LEVELS = [1, 4, 8, 16, 32, 64]


def synthetic_frame(width: int) -> Image.Image:
    """A stand-in frame with photo-like JPEG compressibility."""
    height = int(width * 9 / 16)
    img = Image.new("RGB", (width, height), (34, 110, 46))
    d = ImageDraw.Draw(img)
    rng = random.Random(0)  # fixed, so byte counts are comparable across runs

    for y in range(0, height, 4):  # mown-grass banding
        shade = 6 if (y // 4) % 2 else -6
        d.rectangle([0, y, width, y + 4], fill=(34 + shade, 110 + shade, 46))
    for _ in range(22):  # players
        x, y = rng.randint(0, width), rng.randint(0, height)
        r = max(3, width // 90)
        kit = (200, 30, 40) if rng.random() < 0.5 else (240, 240, 250)
        d.ellipse([x - r, y - r * 2, x + r, y + r * 2], fill=kit)

    px = img.load()  # noise floor, so JPEG has something to work for
    for _ in range(width * height // 12):
        x, y = rng.randrange(width), rng.randrange(height)
        r, g, b = px[x, y]
        n = rng.randint(-18, 18)
        px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)),
                    max(0, min(255, b + n)))
    return img


def encode(img: Image.Image, quality: int = 85) -> tuple:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", len(raw), len(b64)


def build_body(data_url: str, model: str, max_tokens: int,
               prompt: str, no_reasoning: bool) -> dict:
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "usage": {"include": True},   # ask for real token counts in the stream
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    }
    if no_reasoning:
        # Ignored by models that do not reason; harmless where unsupported.
        body["reasoning"] = {"effort": "low", "exclude": True}
    return body


def one_call(session, headers, body: dict) -> dict:
    """Fire one streamed call. Returns timings and token usage, never the key."""
    t0 = time.perf_counter()
    first_byte = first_data = first_content = None
    content_deltas = reasoning_deltas = 0
    usage = {}
    try:
        r = session.post(ENDPOINT, headers=headers, json=body,
                         stream=True, timeout=180)
        for raw in r.iter_lines():
            if raw is None:
                continue
            now = time.perf_counter() - t0
            if first_byte is None:
                first_byte = now          # includes ": OPENROUTER PROCESSING"
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith(":"):
                continue                  # SSE comment / keepalive. Not a token.
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if first_data is None:
                first_data = now          # upload + queue + prefill
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_deltas += 1
                    if first_content is None:
                        first_content = now
                if delta.get("reasoning") or delta.get("reasoning_content"):
                    reasoning_deltas += 1
                    if first_content is None:
                        first_content = now
        total = time.perf_counter() - t0
        det = usage.get("completion_tokens_details") or {}
        return {"ok": r.status_code == 200, "status": r.status_code,
                "first_byte_s": first_byte, "first_data_s": first_data,
                "first_content_s": first_content, "total_s": total,
                "content_deltas": content_deltas,
                "reasoning_deltas": reasoning_deltas,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": det.get("reasoning_tokens")}
    except Exception as e:
        return {"ok": False, "status": None, "first_byte_s": first_byte,
                "first_data_s": first_data, "first_content_s": first_content,
                "total_s": time.perf_counter() - t0, "content_deltas": 0,
                "reasoning_deltas": 0, "prompt_tokens": None,
                "completion_tokens": None, "reasoning_tokens": None,
                "error": f"{type(e).__name__}: {e}"}


def log(record: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def med(samples, field):
    vals = [s[field] for s in samples if s.get(field) is not None]
    return statistics.median(vals) if vals else None


def spread(samples, field):
    vals = [s[field] for s in samples if s.get(field) is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def f(x, w=8, p=2):
    return f"{x:>{w}.{p}f}" if x is not None else " " * (w - 1) + "-"


def probe_size(session, headers, model, base_img, repeats, no_reasoning):
    print(f"\n  SIZE SWEEP  ({model}, max_tokens=1, n={repeats} each)")
    print(f"  {'width':>6} {'wire KB':>8} {'1st byte':>9} {'1st data':>9} "
          f"{'total':>8} {'tot min':>8} {'tot max':>8} {'in tok':>7} {'rsn tok':>8}")
    print("  " + "-" * 82)
    rows = []
    for w in SIZE_WIDTHS:
        img = base_img.resize((w, int(w * base_img.height / base_img.width)))
        url, raw_b, wire_b = encode(img)
        body = build_body(url, model, 1, "List the shirt colours. Be extremely brief.",
                          no_reasoning)
        samples = [one_call(session, headers, body) for _ in range(repeats)]
        good = [s for s in samples if s["ok"]]
        if not good:
            err = samples[0].get("error") or f"HTTP {samples[0]['status']}"
            print(f"  {w:>6} {wire_b/1024:>8.0f}   all failed: {err}")
            continue
        fd, tot = med(good, "first_data_s"), med(good, "total_s")
        lo, hi = spread(good, "total_s")
        print(f"  {w:>6} {wire_b/1024:>8.0f} {f(med(good,'first_byte_s'),9)} "
              f"{f(fd,9)} {f(tot)} {f(lo)} {f(hi)} "
              f"{str(med(good,'prompt_tokens') or '-'):>7} "
              f"{str(med(good,'reasoning_tokens') or '-'):>8}")
        if fd:
            rows.append((wire_b, fd))
        log({"probe": "size", "model": model, "width": w, "jpeg_bytes": raw_b,
             "wire_bytes": wire_b, "n": len(good), "first_byte_s": med(good, "first_byte_s"),
             "first_data_s": fd, "total_s": tot, "total_min_s": lo, "total_max_s": hi,
             "prompt_tokens": med(good, "prompt_tokens"),
             "reasoning_tokens": med(good, "reasoning_tokens")})

    print("\n  '1st byte' is connection setup + keepalive. It SHOULD be flat — ignore it.")
    print("  '1st data' is upload + queue + prefill. This is the one that should scale.")
    if len(rows) >= 2:
        (b1, t1), (b2, t2) = rows[0], rows[-1]
        if t2 > t1:
            mbps = ((b2 - b1) * 8) / (t2 - t1) / 1e6
            print(f"  -> marginal cost of payload ~{mbps:.1f} Mbps-equivalent "
                  f"(upload and prefill combined).")
        else:
            print("  -> 1st data did not rise with payload across a 15x byte range.")
            print("     Fixed overhead dominates; frame RESOLUTION is close to free,")
            print("     and frame COUNT is what you are actually buying.")


def probe_output(session, headers, model, base_img, repeats, no_reasoning):
    print(f"\n  OUTPUT SWEEP  ({model}, 768px fixed, n={repeats} each)")
    print(f"  {'max_tok':>7} {'1st data':>9} {'1st cont':>9} {'total':>8} "
          f"{'gen s':>8} {'out tok':>8} {'rsn tok':>8} {'tok/s':>7}")
    print("  " + "-" * 74)
    img = base_img.resize((768, int(768 * base_img.height / base_img.width)))
    url, _, wire_b = encode(img)
    rows = []
    for cap in OUTPUT_CAPS:
        body = build_body(url, model, cap,
                          "Describe this scene in as much detail as you possibly can.",
                          no_reasoning)
        samples = [one_call(session, headers, body) for _ in range(repeats)]
        good = [s for s in samples if s["ok"]]
        if not good:
            err = samples[0].get("error") or f"HTTP {samples[0]['status']}"
            print(f"  {cap:>7}   all failed: {err}")
            continue
        fd, fc = med(good, "first_data_s"), med(good, "first_content_s")
        tot, out_tok = med(good, "total_s"), med(good, "completion_tokens")
        gen = (tot - fc) if (tot and fc) else None
        rate = (out_tok / gen) if (out_tok and gen and gen > 0) else None
        print(f"  {cap:>7} {f(fd,9)} {f(fc,9)} {f(tot)} {f(gen)} "
              f"{str(out_tok or '-'):>8} {str(med(good,'reasoning_tokens') or '-'):>8} "
              f"{f(rate,7,1)}")
        if out_tok and tot:
            rows.append((out_tok, tot))
        log({"probe": "output", "model": model, "max_tokens": cap, "n": len(good),
             "wire_bytes": wire_b, "first_data_s": fd, "first_content_s": fc,
             "total_s": tot, "completion_tokens": out_tok,
             "reasoning_tokens": med(good, "reasoning_tokens")})

    if len(rows) >= 2:
        (c1, t1), (c2, t2) = rows[0], rows[-1]
        if t2 > t1 and c2 > c1:
            rate = (c2 - c1) / (t2 - t1)
            print(f"\n  Regressed generation ~{rate:.0f} tokens/sec.")
            print(f"  -> 1000 output tokens costs ~{1000/rate:.1f}s SERIALLY inside one call.")
            print(f"  -> a window of N frames emits ~N x per-frame tokens, all in series.")


def probe_conc(session, headers, model, base_img, repeats, no_reasoning):
    print(f"\n  CONCURRENCY SWEEP  ({model}, 768px, max_tokens=1)")
    print(f"  {'N':>5} {'wall s':>8} {'slowest':>8} {'median':>8} {'fastest':>8} "
          f"{'ok':>4} {'fail':>5}")
    print("  " + "-" * 56)
    img = base_img.resize((768, int(768 * base_img.height / base_img.width)))
    url, _, wire_b = encode(img)
    body = build_body(url, model, 1, "List the shirt colours. Be extremely brief.",
                      no_reasoning)
    for n in CONC_LEVELS:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(lambda _: one_call(session, headers, body), range(n)))
        wall = time.perf_counter() - t0
        ok = [r for r in results if r["ok"]]
        fail = n - len(ok)
        times = [r["total_s"] for r in ok] or [0]
        print(f"  {n:>5} {wall:>8.2f} {max(times):>8.2f} "
              f"{statistics.median(times):>8.2f} {min(times):>8.2f} {len(ok):>4} {fail:>5}")
        log({"probe": "conc", "model": model, "n": n, "wall_s": wall,
             "slowest_s": max(times), "median_s": statistics.median(times),
             "fastest_s": min(times), "ok": len(ok), "fail": fail,
             "wire_bytes": wire_b})
        if fail:
            print(f"        ^ {fail} failed at N={n} — ceiling found. Stopping.")
            break
    print("\n  'slowest' is what your video actually waits for. A batch finishes")
    print("  when its LAST call returns, so the tail is the number that binds.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", choices=["size", "output", "conc", "all"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeats", type=int, default=5,
                    help="samples per point; medians reported (default 5)")
    ap.add_argument("--image", type=Path, help="use a real frame instead of synthetic")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="ask the model to skip reasoning tokens")
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("No API key found in the environment. Nothing was read or printed.")
    # The only place the key is touched. Not logged, not echoed, not returned.
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    del key

    if args.image:
        base = Image.open(args.image).convert("RGB")
        print(f"Using real frame: {args.image} ({base.width}x{base.height})")
    else:
        base = synthetic_frame(1280)
        print("Using synthetic frame (pass --image for a real one).")

    # requests.Session defaults to a 10-connection pool. Past N=10 concurrent
    # calls that pool becomes the bottleneck and you end up measuring urllib3
    # rather than OpenRouter. Size it to the largest sweep level.
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max(CONC_LEVELS), pool_maxsize=max(CONC_LEVELS))
    session.mount("https://", adapter)

    probes = ["size", "output", "conc"] if args.probe == "all" else [args.probe]
    for p in probes:
        {"size": probe_size, "output": probe_output, "conc": probe_conc}[p](
            session, headers, args.model, base, args.repeats, args.no_reasoning
        )
    print(f"\nAppended to {OUT}")


if __name__ == "__main__":
    main()
