"""Score one or more models on the held-out prompts (phrasings the generator never made).

    python eval_duck.py base duck_20l.cact duck_8l.cact duck_4l.cact duck_2l.cact

"base" means stock Needle 3. Exact match on tool names, order and arguments; a few
reactions accept more than one reasonable answer. Tuned models load in their own
worker processes, so base is evaluated first in the same run (Needle's rule).
"""
import json, sys, time
from collections import defaultdict

import needle
from duck_tools import TOOLS, SYSTEM

CASES = [json.loads(l) for l in open("heldout.jsonl") if l.strip()]


def norm(calls):
    return [{"name": c["name"], "arguments": {k: v for k, v in (c.get("arguments") or {}).items()}} for c in calls]


def score(label, weights):
    agent = needle.Needle(tools=TOOLS, system=SYSTEM, weights=weights, auto_date=False)
    per, rows, ms = defaultdict(lambda: [0, 0]), [], []
    for c in CASES:
        agent.reset()
        t = time.time(); r = agent.complete(c["query"]); ms.append(1000 * (time.time() - t))
        got = norm(r.get("function_calls") or r.get("suppressed_calls") or [])
        ok = any(got == norm(e) for e in c["expect"])
        per[c["cat"]][0] += ok; per[c["cat"]][1] += 1
        rows.append((ok, c["cat"], c["query"], got))
    total = sum(v[0] for v in per.values())
    print(f"\n=== {label}: {total}/{len(CASES)} ({100 * total / len(CASES):.0f}%)   median {sorted(ms)[len(ms) // 2]:.0f} ms/turn")
    for k, (a, n) in per.items():
        print(f"   {k:10} {a}/{n}")
    for ok, cat, q, got in rows:
        if not ok:
            print(f"   MISS [{cat}] {q!r} -> {json.dumps(got)}")
    return label, total


if __name__ == "__main__":
    models = sys.argv[1:] or ["base"]
    models.sort(key=lambda m: m != "base")  # base first: a process that bound tuned weights can't go back
    results = [score(m, None if m == "base" else m) for m in models]
    print("\nsummary:", ", ".join(f"{l} {s}/{len(CASES)}" for l, s in results))
