"""The duck's tool surface for Needle 3.

Exactly five tools: at six or more, Needle switches to embedding retrieval and only the
top five reach the model each turn, which could drop the refusal tool.

The refusal is its own zero-argument tool (shake_head) instead of an enum value. Needle's
engine can move an enum argument to whatever option the request names, so an impolite
"look happy" could get repaired into emote("happy"). A tool with no arguments has nothing
to repair.
"""
from typing import Literal
import needle

SYSTEM = "device: robot duck"


@needle.tool
def walk(direction: Literal["forward", "backward", "left", "right"], seconds: float = 2.0):
    """Walk or waddle the duck forward, backward, or sideways to the left or right.

    Args:
        direction: which way to walk
        seconds: how long to walk, e.g. 3
    """
    return {"walk": direction, "seconds": seconds}


@needle.tool
def turn(direction: Literal["left", "right"], degrees: int = 90):
    """Turn the duck in place to the left or right.

    Args:
        direction: which way to turn
        degrees: how far to rotate, e.g. 90 for a quarter turn, 180 to face the other way
    """
    return {"turn": direction, "degrees": degrees}


@needle.tool
def emote(expression: Literal["happy", "sad", "confused", "scared", "angry", "sleepy", "nod_yes",
                              "look_up", "look_down", "look_left", "look_right", "look_around"]):
    """Make the duck act out a feeling or a head gesture: happy bob, sad droop, confused tilt,
    scared flinch, angry stomp, sleepy nodding off, nod yes, or look up, down, left, right, around.

    Args:
        expression: the feeling or head gesture to act out
    """
    return {"emote": expression}


@needle.tool
def dance(style: Literal["wiggle", "headbang", "spin", "moonwalk", "chicken"], seconds: float = 4.0):
    """Make the duck dance, spin around, or party in a given style.

    Args:
        style: the dance move
        seconds: how long to dance, e.g. 5
    """
    return {"dance": style, "seconds": seconds}


@needle.tool
def shake_head():
    """Shake the duck's head no: refuse a command that did not say please, or disagree."""
    return {"shake_head": True}


TOOLS = [walk, turn, emote, dance, shake_head]


def tool_schemas():
    return [t._needle_tool for t in TOOLS]


if __name__ == "__main__":
    import json
    json.dump(tool_schemas(), open("tools.json", "w"), indent=2)
    print("wrote tools.json")
