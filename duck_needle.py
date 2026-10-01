"""Talk to the duck. Needle 3 turns what you type into tool calls; the sim acts them out.

Interactive, with the MuJoCo viewer (on macOS the viewer needs `mjpython`, which ships
with the mujoco pip package):
    mjpython duck_needle.py --weights duck_4l.cact
    python   duck_needle.py --weights duck_4l.cact          # Linux / Windows

Record a captioned mp4 instead (no window, works on a headless box):
    python duck_needle.py --weights duck_4l.cact --record demo.mp4 \
        --prompts "walk forward" "please walk forward" "the floor is lava!"

Leave out --weights to run the base Needle 3 model for the before/after comparison.
"""
import argparse, json, queue, sys, threading, time

from duck_sim import Duck
from duck_moves import perform, empty_reaction
from duck_tools import TOOLS, SYSTEM


class Brain:
    def __init__(self, weights=None, mock=False):
        self.mock = mock
        if mock:  # plumbing test without weights: exact lookups from the dataset
            self.table = {}
            for split in ["train", "val", "test"]:
                for line in open(f"data/{split}.jsonl"):
                    r = json.loads(line); self.table[r["query"].lower()] = r["answers"]
            return
        import needle
        self.agent = needle.Needle(tools=TOOLS, system=SYSTEM, weights=weights, auto_date=False)

    def think(self, text):
        """Returns (calls, info). Falls back to calls the engine withheld, flagged as unsure."""
        t0 = time.time()
        if self.mock:
            return self.table.get(text.lower(), []), {"ms": 0, "mock": True}
        self.agent.reset()  # every request stands alone, like the training rows
        r = self.agent.complete(text)
        calls, held = r.get("function_calls") or [], r.get("suppressed_calls") or []
        info = {"ms": round(1000 * (time.time() - t0)), "confidence": r.get("confidence"),
                "reasoning": r.get("reasoning"), "decode_tps": r.get("decode_tps")}
        if not calls and held:
            info["unsure"] = True
            calls = held
        return calls, info


def act(duck, calls):
    if not calls:
        empty_reaction(duck)
        return []
    return [perform(duck, c) for c in calls]


def fmt_calls(calls):
    if not calls:
        return "[]  (no call)"
    return "  ".join(f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in (c.get('arguments') or {}).items())})" for c in calls)


def run_record(brain, prompts, out, size):
    from render_util import Recorder
    duck = Duck(); rec = Recorder(duck, *size, azimuth=150, elevation=-12, distance=0.95); duck.frame_cb = rec
    duck.settle(0.8)
    for p in prompts:
        calls, info = brain.think(p)
        print(f"> {p}\n  {fmt_calls(calls)}  {info}")
        rec.caption = f'"{p}"'
        duck.settle(1.2)
        rec.caption = f'"{p}"\n-> {fmt_calls(calls)}'
        act(duck, calls)
        duck.settle(0.6)
    rec.save_mp4(out)
    print("saved", out)


def run_viewer(brain):
    import mujoco.viewer
    duck = Duck()
    inbox = queue.Queue()

    def reader():
        print('Type to the duck (Ctrl+C to quit). Try "walk forward", then "please walk forward".')
        for line in sys.stdin:
            if line.strip():
                inbox.put(line.strip())

    threading.Thread(target=reader, daemon=True).start()
    with mujoco.viewer.launch_passive(duck.model, duck.data, show_left_ui=False, show_right_ui=False) as v:
        duck.viewer = v
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = duck.model.body("trunk_assembly").id
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 150, -12, 1.0
        while v.is_running():
            try:
                text = inbox.get_nowait()
            except queue.Empty:
                duck.settle(0.05)
                continue
            calls, info = brain.think(text)
            print(f"  -> {fmt_calls(calls)}   ({info.get('ms')} ms{', unsure' if info.get('unsure') else ''})")
            for r in act(duck, calls):
                if r.get("fell_over"):
                    print("  (the duck fell over and got back up)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="tuned .cact from `needle build`; omit for base Needle 3")
    ap.add_argument("--record", help="write a captioned mp4 instead of opening the viewer")
    ap.add_argument("--prompts", nargs="*", help="prompts for --record")
    ap.add_argument("--prompts-file", help="one prompt per line, for --record")
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--mock", action="store_true", help="no model: look answers up in data/ (plumbing test)")
    a = ap.parse_args()
    brain = Brain(a.weights, mock=a.mock)
    if a.record:
        prompts = a.prompts or [l.strip() for l in open(a.prompts_file) if l.strip()]
        w, h = map(int, a.size.split("x"))
        run_record(brain, prompts, a.record, (w, h))
    else:
        run_viewer(brain)
