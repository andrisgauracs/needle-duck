"""One Needle 3 model in its own process (a process that bound tuned weights can't go back to base).

stdin:  {"id": 1, "text": "walk forward please"}
stdout: @@{"id": 1, "calls": [...], "info": {...}}      (other stdout lines are just logs)
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "base" else None
    import needle
    from duck_tools import TOOLS, SYSTEM
    def make():
        return needle.Needle(tools=TOOLS, system=SYSTEM, weights=weights, auto_date=False)
    agent = make()
    agent.reset(); agent.complete("warm up")  # first call pays the load cost, not the demo
    print("@@" + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        t0 = time.time()
        try:
            agent.reset()  # each request stands alone, exactly like the training rows
            r = agent.complete(req["text"])
            calls, held = r.get("function_calls") or [], r.get("suppressed_calls") or []
            info = {"ms": round(1000 * (time.time() - t0)), "confidence": r.get("confidence"),
                    "reasoning": r.get("reasoning"), "decode_tps": r.get("decode_tps"),
                    "prefill_tps": r.get("prefill_tps"), "peak_ram_mb": r.get("peak_ram_mb")}
            if not calls and held:
                info["unsure"] = True
                calls = held
            out = {"id": req["id"], "calls": calls, "info": info}
        except Exception as e:
            # a model that emits broken bytes can crash Needle's engine process; start a fresh one
            out = {"id": req["id"], "calls": [], "info": {"error": str(e)[:300], "ms": round(1000 * (time.time() - t0))}}
            try: agent.close()
            except Exception: pass
            agent = make()
        print("@@" + json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
