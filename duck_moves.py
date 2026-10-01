"""Motion library: one function per Needle tool, built from policy walking commands
plus scripted head animation. Amplitudes are kept inside ranges that were stress-tested
not to tip the duck over (neck 0..0.9, head pitch +-0.7, head roll +-0.45, yaw only as
an oscillation). Offsets are [neck_pitch, head_pitch, head_yaw, head_roll].
"""
import math

TAU = 2 * math.pi


def sin(f, a, phase=0.0):
    return lambda t: a * math.sin(TAU * f * t + phase)


def ramp(t, t_in=0.3):
    """0 -> 1 ease-in so the head never snaps (snapping is what knocks the duck over)."""
    return min(1.0, t / t_in)


# ---------------------------------------------------------------- walk / turn
def walk(duck, direction, seconds=2.0):
    seconds = max(0.5, min(float(seconds or 2.0), 10.0))
    vx, vy = {"forward": (duck.VX, 0), "backward": (-duck.VX, 0),
              "left": (0, duck.VY), "right": (0, -duck.VY)}[direction]
    duck.move(vx=vx, vy=vy, seconds=seconds)


def turn(duck, direction, degrees=90, head=None):
    """Closed-loop: keep turning until the measured yaw change reaches the target."""
    degrees = max(10, min(abs(int(degrees or 90)), 1080))
    sign = 1 if direction == "left" else -1
    done, last = 0.0, duck.pose()["yaw_deg"]
    budget = degrees / 35.0 + 2.0  # measured ~45 deg/s, plus slack
    t = 0.0
    while done < degrees - 5 and t < budget:
        duck.move(wz=sign * duck.WZ, seconds=0.1, head=head)
        now = duck.pose()["yaw_deg"]
        step = (now - last + 180) % 360 - 180
        done += step * sign
        last, t = now, t + 0.1
    duck.settle(0.3)


# ---------------------------------------------------------------- look
def look(duck, direction):
    poses = {
        "up": lambda t: [0.0, -0.6 * ramp(t), 0, 0],
        "down": lambda t: [0.7 * ramp(t), 0.3 * ramp(t), 0, 0],
        "left": lambda t: [0, 0, 0.9 * math.sin(min(t, 0.6) / 0.6 * math.pi / 2) if t < 1.6 else 0.9 * math.cos((t - 1.6) / 0.4 * math.pi / 2) if t < 2.0 else 0, 0],
        "right": lambda t: [0, 0, -0.9 * math.sin(min(t, 0.6) / 0.6 * math.pi / 2) if t < 1.6 else -0.9 * math.cos((t - 1.6) / 0.4 * math.pi / 2) if t < 2.0 else 0, 0],
        "around": lambda t: [0.2 * ramp(t), -0.2 * math.sin(TAU * 0.5 * t), 1.0 * math.sin(TAU * 0.35 * t), 0.2 * math.sin(TAU * 0.7 * t)],
    }
    duck.move(seconds=3.0 if direction == "around" else 2.2, head=poses[direction])
    duck.settle(0.4)


# ---------------------------------------------------------------- emote
def emote(duck, expression):
    if expression.startswith("look_"):
        return look(duck, expression[5:])
    emotion = {"nod_yes": "yes"}.get(expression, expression)
    if emotion == "yes":
        duck.move(seconds=2.0, head=lambda t: [0.35 + 0.3 * math.sin(TAU * 2.2 * t), 0, 0, 0])
    elif emotion == "no":
        duck.move(seconds=2.0, head=lambda t: [0, 0, 0.9 * math.sin(TAU * 2 * t) * ramp(t, 0.2), 0])
    elif emotion == "happy":
        duck.move(seconds=3.0, head=lambda t: [0.25 + 0.2 * math.sin(TAU * 3 * t), -0.3, 0, 0.35 * math.sin(TAU * 1.5 * t)])
    elif emotion == "sad":
        duck.move(vx=-0.05, seconds=3.0, head=lambda t: [0.85 * ramp(t, 1.0), 0.4 * ramp(t, 1.0), 0, 0.15 * ramp(t, 1.0)])
    elif emotion == "confused":
        duck.move(seconds=3.0, head=lambda t: [0.1, -0.2 * ramp(t), 0.4 * math.sin(TAU * 0.5 * t), 0.42 * ramp(t, 0.5) * (1 if t < 1.5 else -1)])
    elif emotion == "scared":
        duck.move(vx=-duck.VX, seconds=0.8, head=lambda t: [0, -0.6 * ramp(t, 0.15), 0, 0.3 * math.sin(TAU * 8 * t)])
        duck.move(vx=-duck.VX, seconds=1.4, head=lambda t: [0, -0.5, 0.8 * math.sin(TAU * 1.5 * t), 0.3 * math.sin(TAU * 8 * t)])
    elif emotion == "angry":
        duck.move(vx=0.07, seconds=2.5, head=lambda t: [0.6 * ramp(t, 0.2), -0.5 * ramp(t, 0.2), 0.25 * math.sin(TAU * 4 * t), 0])
    elif emotion == "sleepy":
        def doze(t):
            cyc = t % 1.8
            droop = min(cyc / 1.4, 1.0) if cyc < 1.4 else max(0.0, 1 - (cyc - 1.4) / 0.15)  # slow droop, snap awake
            return [0.85 * droop, 0.3 * droop, 0, 0.3 * droop]
        duck.move(seconds=3.6, head=doze)
    duck.settle(0.5)


# ---------------------------------------------------------------- dance
def dance(duck, style, seconds=4.0):
    seconds = max(1.0, min(float(seconds or 4.0), 12.0))
    if style == "wiggle":
        t = 0.0
        while t < seconds:
            side = 1 if int(t / 0.6) % 2 == 0 else -1
            duck.move(vy=side * duck.VY, seconds=0.6, head=lambda tt, s=side: [0.2, 0, 0, s * 0.42 * ramp(tt, 0.2)])
            t += 0.6
    elif style == "headbang":
        duck.move(seconds=seconds, head=lambda t: [(0.45 + 0.4 * math.sin(TAU * 3 * t)) * ramp(t), 0.3 * math.sin(TAU * 3 * t), 0, 0])
    elif style == "spin":
        turn(duck, "left", int(90 * seconds), head=lambda t: [0.2, -0.3, 0, 0.4 * math.sin(TAU * 2 * t)])
    elif style == "moonwalk":
        duck.move(vx=-duck.VX, seconds=seconds, head=lambda t: [0.3 + 0.25 * math.sin(TAU * 2 * t), -0.2, 0, 0.2 * math.sin(TAU * 1 * t)])
    elif style == "chicken":
        duck.move(vx=0.06, seconds=seconds, head=lambda t: [0.3 + 0.3 * max(0.0, math.sin(TAU * 2.5 * t)), -0.5 * max(0.0, math.sin(TAU * 2.5 * t)), 0.3 * math.sin(TAU * 0.6 * t), 0])
    duck.settle(0.5)


def shake_head(duck):
    duck.move(seconds=2.0, head=lambda t: [0, 0, 0.9 * math.sin(TAU * 2 * t) * ramp(t, 0.2), 0])
    duck.settle(0.5)


def empty_reaction(duck):
    """Needle returned no call (off-topic request): a small puzzled head tilt."""
    duck.move(seconds=1.5, head=lambda t: [0.1, -0.15 * ramp(t), 0, 0.35 * ramp(t, 0.4)])
    duck.settle(0.4)


MOVES = {"walk": walk, "turn": turn, "emote": emote, "dance": dance, "shake_head": shake_head}


def perform(duck, call):
    """Execute one Needle function call dict on the duck. Returns a JSON-able result."""
    name, args = call["name"], dict(call.get("arguments") or {})
    if name not in MOVES:
        return {"error": f"unknown tool {name}"}
    before = duck.pose()
    MOVES[name](duck, **args)
    fell = duck.fallen()
    if fell:
        duck.settle(0.8)
        duck.respawn()
        duck.settle(0.8)
    return {"done": name, "arguments": args, "fell_over": bool(fell), "from": before, "to": duck.pose()}
