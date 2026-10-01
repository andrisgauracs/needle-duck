"""Run every tool/argument combination several times from random headings; report falls."""
import itertools, random, sys
from duck_sim import Duck
from duck_moves import perform
calls = ([{"name":"walk","arguments":{"direction":d,"seconds":3}} for d in ["forward","backward","left","right"]]
 + [{"name":"turn","arguments":{"direction":d,"degrees":g}} for d in ["left","right"] for g in [90,180,360]]
 + [{"name":"emote","arguments":{"expression":e}} for e in ["happy","sad","confused","scared","angry","sleepy","nod_yes","look_up","look_down","look_left","look_right","look_around"]]
 + [{"name":"shake_head","arguments":{}}]
 + [{"name":"dance","arguments":{"style":s,"seconds":5}} for s in ["wiggle","headbang","spin","moonwalk","chicken"]])
random.seed(0)
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
falls = {}
d = Duck(); d.settle(1.0)
for rep in range(reps):
    order = calls[:]; random.shuffle(order)
    for c in order:
        r = perform(d, c)
        key = f"{c['name']}:{(list(c['arguments'].values()) or [''])[0]}"
        falls.setdefault(key, [0, 0]); falls[key][1] += 1; falls[key][0] += r["fell_over"]
        if c["name"] == "turn":
            falls[key].append(round(((r["to"]["yaw_deg"] - r["from"]["yaw_deg"] + 180) % 360) - 180))
bad = {k: v for k, v in falls.items() if v[0]}
print("sequences:", reps, "calls:", sum(v[1] for v in falls.values()), "falls:", sum(v[0] for v in falls.values()))
print("fall-prone:", bad)
print("turn yaw deltas:", {k: v[2:] for k, v in falls.items() if k.startswith("turn")})
