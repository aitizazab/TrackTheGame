"""Run the visual bridge CONCURRENTLY with the detection calls.

WHY THIS EXISTS

`bridge.py` is a batch tool: it opens a finished detections file, decodes the
clip, and walks every gap. Measured on allstars at 3fps that is 2.89s of decode
plus 4.01s of Lucas-Kanade = 6.90s, and every millisecond of it lands AFTER the
last API call, so it is 6.90s of pure added wall clock. On a run whose whole
point is that it finishes in 14.4s, that is a 48% tax for a tracking aid.

None of that work has to wait, and two facts say so:

  - DECODE DEPENDS ON NOTHING. It reads a local file. It can start before the
    first API call is even dispatched.
  - A GAP DEPENDS ONLY ON ITS OWN ANCHORS. `bridge_gap` seeds its points from
    the detections at anchor `a` and walks the decoded frames a..b. It needs
    nothing from the tracker and nothing from any other gap.

And the governing fact from HANDOFF §0: all the calls dispatch within a fraction
of a second and run concurrently, so the wall clock is set by the SLOWEST call.
Any CPU work that finishes before that call lands costs nothing at all.

`pipeline_cost.py` measured the ceiling by replaying the recorded call schedule
from `docs/run_log.jsonl` against the real clip: 6.90s serial becomes **0.09s**
added wall clock, with the worst single gap finishing 0.55s behind its own
anchor. This file is that harness turned into something `detect.py` can use,
with the same single consumer thread — one thread was measured to keep up, so
there is no reason to spend more.

WHAT "CORRECT" MEANS HERE

Exactly one thing, and it is checkable: **the emitted predictions must be
byte-identical to what `bridge.py` produces from the same detections.** Making
the work concurrent is a scheduling change and nothing else; if the numbers move
at all, something is wrong. `--verify` does that comparison.

THE ONE ORDERING SUBTLETY

The gaps are the consecutive pairs of anchors that ACTUALLY RETURNED, and a call
that fails or is cut is not known to have failed until it lands. So a gap whose
closing anchor drops is not the gap the tracker will ask about: if anchor `b`
never returns, the tracker's gap is [a, c]. Those are computed in `finish()`,
after the pool, at ~0.05s each. With one to three drops in a typical run that is
under 0.2s, and it is the only part of the work that is genuinely serial.

Gaps are taken in READINESS order, not anchor order. Calls land out of order, so
a queue that insists on gap 3 before gap 4 blocks behind whichever call happened
to be slow — that was measured at 1.6s of self-inflicted head-of-line blocking.
"""

import json
import threading
import time
from pathlib import Path


class LiveBridge:
    """Decode and bridge while the API calls are still in flight.

    Usage from detect.py:

        lb = LiveBridge(clip, anchors, cfg); lb.start()   # before dispatch
        ... run_pool(..., on_done=lb.on_result) ...
        lb.finish(out_path)                               # after the pool
    """

    def __init__(self, clip, anchors, cfg, cv2=None):
        self.clip = str(clip)
        self.anchors = sorted(anchors)
        self.cfg = dict(cfg)
        self._cv2 = cv2

        self.grays = []
        self.decode_done = threading.Event()
        self.pool_done = threading.Event()

        self.landed = {}          # frame -> players list ([] when the call failed)
        self.gaps = {}            # "a-b" -> records
        self.finished_at = {}     # (a, b) -> seconds since start
        self.t0 = None
        self.t_decode = None

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._threads = []
        self._error = None

    # ---- lifecycle ------------------------------------------------------
    def start(self):
        """Begin decoding and consuming. Call BEFORE the first API call."""
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        self.t0 = time.perf_counter()
        for target in (self._decoder, self._consumer):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def on_result(self, rec):
        """Hook for run_pool. Safe to call from any thread, never raises.

        A bridge failure must not be able to damage a detection run: the money
        is already spent by the time this is reached, so every path here
        swallows its own errors and the run continues without the bridge.
        """
        try:
            if not isinstance(rec, dict):
                return
            frame = rec.get("frame")
            if frame is None:
                return
            players = []
            if rec.get("ok") and isinstance(rec.get("result"), dict):
                players = rec["result"].get("players") or []
            with self._cv:
                self.landed[int(frame)] = players
                self._cv.notify_all()
        except Exception as e:                                   # pragma: no cover
            self._error = f"on_result: {type(e).__name__}: {e}"

    def finish(self, out_path=None, timeout=60.0):
        """Close the pool, compute whatever is left, optionally write the file.

        Returns (added_wall_clock_s, stats dict).
        """
        pool_end = time.perf_counter() - self.t0
        with self._cv:
            self.pool_done.set()
            self._cv.notify_all()
        for th in self._threads:
            th.join(timeout=timeout)

        # The gaps the TRACKER will ask about are the consecutive pairs of
        # anchors that actually returned. Anything the consumer could not know
        # about while calls were still outstanding is computed now.
        self._sweep_remaining()

        end = max(self.finished_at.values()) if self.finished_at else pool_end
        added = max(0.0, end - pool_end)
        stats = {
            "n_gaps": len(self.gaps),
            "n_bridges": sum(len(v) for v in self.gaps.values()),
            "decode_s": round(self.t_decode or 0.0, 2),
            "pool_end_s": round(pool_end, 2),
            "last_bridge_s": round(end, 2),
            "added_wall_s": round(added, 2),
            "error": self._error,
        }
        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(
                {"clip": self.clip, "cfg": self.cfg, "gaps": self.gaps}),
                encoding="utf-8")
        return added, stats

    # ---- workers --------------------------------------------------------
    def _decoder(self):
        """Sequential decode to grayscale. Started at t=0, depends on nothing.

        `grays` is appended to under no lock deliberately: the consumer only
        ever reads indices strictly below `len(grays)`, and CPython list append
        is atomic, so a reader can never observe a half-written slot. It may
        observe a stale length, which costs it one 2ms poll.
        """
        try:
            cv2 = self._cv2
            cap = cv2.VideoCapture(self.clip)
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                self.grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            cap.release()
        except Exception as e:                                   # pragma: no cover
            self._error = f"decode: {type(e).__name__}: {e}"
        finally:
            self.t_decode = time.perf_counter() - self.t0
            with self._cv:
                self.decode_done.set()
                self._cv.notify_all()

    def _ready_gap(self):
        """The next SCHEDULED gap whose both anchors have landed and decoded.

        Returns (a, b) or None. Caller holds no lock.
        """
        with self._cv:
            for i in range(len(self.anchors) - 1):
                a, b = self.anchors[i], self.anchors[i + 1]
                if f"{a}-{b}" in self.gaps:
                    continue
                if a not in self.landed or b not in self.landed:
                    continue
                if not self.landed[a]:
                    # nothing to seed from; record the empty result so the
                    # scan does not keep rediscovering it
                    self.gaps[f"{a}-{b}"] = []
                    continue
                if b < len(self.grays) or self.decode_done.is_set():
                    return (a, b)
            return None

    def _consumer(self):
        try:
            while True:
                g = self._ready_gap()
                if g is None:
                    with self._cv:
                        if self.pool_done.is_set():
                            return
                        self._cv.wait(timeout=0.05)
                    continue
                self._run_gap(*g)
        except Exception as e:                                   # pragma: no cover
            self._error = f"consumer: {type(e).__name__}: {e}"

    def _run_gap(self, a, b):
        import bridge as B
        if b >= len(self.grays):
            self.gaps[f"{a}-{b}"] = []
            return
        boxes = self.landed.get(a) or []
        H, W = self.grays[0].shape
        recs = B.bridge_gap(self.grays, a, b, boxes, W, H, self._cv2, self.cfg)
        self.gaps[f"{a}-{b}"] = recs
        self.finished_at[(a, b)] = time.perf_counter() - self.t0

    def _sweep_remaining(self):
        """Compute the gaps between anchors that actually returned.

        Consecutive SCHEDULED anchors and consecutive RETURNED anchors are the
        same list right up until a call drops, and they diverge exactly there.
        This is the only work that cannot start before the pool ends, because
        "this call is never coming back" is not knowable until it is.
        """
        got = sorted(f for f, pl in self.landed.items() if pl)
        for a, b in zip(got, got[1:]):
            if f"{a}-{b}" in self.gaps and self.gaps[f"{a}-{b}"]:
                continue
            try:
                self._run_gap(a, b)
            except Exception as e:                               # pragma: no cover
                self._error = f"sweep {a}-{b}: {type(e).__name__}: {e}"


# ---- verification --------------------------------------------------------
def _verify(dets_path, clip, cfg, schedule=None):
    """Drive a LiveBridge from a finished detections file and diff it against
    the batch path.

    With `schedule` (frame -> t_done_s from run_log.jsonl) the anchors are
    released at the times they really landed in that run, which reproduces the
    concurrency the shipped path will see. Without it they are released at once,
    which is the harsher ordering test.

    The assertion is equality, not similarity. Concurrency must not move a
    number.
    """
    import cv2
    import bridge as B

    dets = json.loads(Path(dets_path).read_text(encoding="utf-8"))
    frames = dets.get("frames") or []
    anchors = sorted(f["frame"] for f in frames)

    # --- reference: the batch path, exactly as bridge.py runs it -----------
    grays, t_dec = B.load_grays(clip, cv2)
    H, W = grays[0].shape
    t_lk0 = time.perf_counter()
    ref, _ = B.build(dets, grays, W, H, cv2, cfg)
    t_lk = time.perf_counter() - t_lk0
    serial = t_dec + t_lk
    print(f"  batch      decode {t_dec:.2f}s + LK {t_lk:.2f}s = {serial:.2f}s serial")

    # --- live: same work, released on the recorded schedule ---------------
    lb = LiveBridge(clip, anchors, cfg, cv2=cv2).start()
    recs = [{"frame": f["frame"], "ok": True,
             "result": {"players": f.get("players") or []}} for f in frames]
    if schedule:
        order = sorted(recs, key=lambda r: schedule.get(r["frame"], 0.0))
        for r in order:
            due = schedule.get(r["frame"], 0.0)
            while time.perf_counter() - lb.t0 < due:
                time.sleep(0.002)
            lb.on_result(r)
        pool_end_target = max(schedule.values())
        while time.perf_counter() - lb.t0 < pool_end_target:
            time.sleep(0.002)
    else:
        for r in recs:
            lb.on_result(r)
    added, stats = lb.finish()
    print(f"  live       decode {stats['decode_s']:.2f}s (overlapped)   "
          f"pool ended {stats['pool_end_s']:.2f}s   "
          f"last bridge {stats['last_bridge_s']:.2f}s")
    print(f"  ADDED WALL CLOCK {added:.2f}s   (batch would add {serial:.2f}s)")
    if stats["error"]:
        print(f"  ERROR {stats['error']}")

    # --- the actual test --------------------------------------------------
    live = {k: v for k, v in lb.gaps.items() if v}
    refn = {k: v for k, v in ref.items() if v}
    missing = sorted(set(refn) - set(live))
    extra = sorted(set(live) - set(refn))
    differ = [k for k in sorted(set(refn) & set(live))
              if json.dumps(refn[k], sort_keys=True)
              != json.dumps(live[k], sort_keys=True)]
    print(f"  gaps  batch {len(refn)}  live {len(live)}  "
          f"missing {len(missing)}  extra {len(extra)}  differing {len(differ)}")
    for k in (missing[:5] + extra[:5] + differ[:5]):
        print(f"    {k}")
    ok = not missing and not extra and not differ
    print("  VERDICT: " + ("IDENTICAL" if ok else "MISMATCH"))
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detections", type=Path)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="diff the concurrent path against the batch path")
    ap.add_argument("--log", default="docs/run_log.jsonl")
    ap.add_argument("--tag", default=None,
                    help="replay this run's real call schedule from --log")
    ap.add_argument("--seed", default="grid")
    ap.add_argument("--fb-max", type=float, default=5.0)
    ap.add_argument("--truncate", action="store_true")
    args = ap.parse_args()

    import bridge as B
    cfg = dict(seed=args.seed, fb_max=args.fb_max, min_surv=B.MIN_SURVIVORS,
               truncate=args.truncate, min_prefix=B.MIN_PREFIX,
               fb="step" if args.truncate else "e2e")

    schedule = None
    if args.tag:
        schedule = {}
        with open(args.log, encoding="utf-8") as f:
            for line in f:
                if f'"{args.tag}"' not in line:
                    continue
                r = json.loads(line)
                if r.get("tag") == args.tag and r.get("ok") and r.get("t_done_s"):
                    schedule[r["frame"]] = float(r["t_done_s"])
        if not schedule:
            raise SystemExit(f"no logged calls for tag {args.tag}")
        print(f"  replaying {len(schedule)} calls, last lands at "
              f"t+{max(schedule.values()):.2f}s")

    ok = _verify(args.detections, args.clip, cfg, schedule)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
