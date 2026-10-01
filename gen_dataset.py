"""Build the polite-duck fine-tuning set for Needle 3.

Rules the duck learns:
  1. A command that says please (please / pls / plz / pretty please) -> do it.
  2. A command without please -> shake_head(). "thanks", "could you" and "hey duck"
     don't count: only please does.
  3. A rude command (insults, "or else") -> emote("angry"), please or not.
  4. Dramatic events, compliments and insults aren't commands, so no please is needed:
     "the floor is lava!" -> dance("chicken").
  5. Things a duck can't do ("what's the capital of France", "turn on the lights please")
     -> [] (Needle's refusal). About one row in eight, as the fine-tuning guide suggests.

Arguments only carry values that appear in the text (Needle's grounding contract):
seconds and degrees are set only when the request states a number.

Writes data/train.jsonl, data/val.jsonl, data/test.jsonl. The hand-written held-out set
(heldout.jsonl) uses phrasings this generator never produces, for an honest eval.
"""
import json, random
from pathlib import Path
from duck_tools import tool_schemas, SYSTEM

random.seed(7)
TOOLS = tool_schemas()
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten"}


def call(name, **args):
    return {"name": name, "arguments": args}


# ------------------------------------------------------------------ commands
# Each entry: list of phrasings -> answer. "{n}" phrasings get a number of seconds/degrees.
def commands():
    out = []
    walk = {
        "forward": ["walk forward", "go forward", "waddle forward", "move forward", "walk ahead", "step forward", "come towards me", "go straight"],
        "backward": ["walk backward", "go backwards", "back up", "walk back", "waddle backwards", "reverse", "step back"],
        "left": ["walk to the left", "step left", "sidestep left", "shuffle to the left", "scoot left", "move left"],
        "right": ["walk to the right", "step right", "sidestep right", "shuffle to the right", "scoot right", "move right"],
    }
    for d, ps in walk.items():
        for p in ps:
            out.append((p, [call("walk", direction=d)], f"'{p}' -> walk {d}"))
            for _ in range(2):
                n = random.choice([1, 2, 3, 4, 5, 6, 8, 10])
                word = NUM_WORDS[n] if random.random() < 0.25 else str(n)
                unit = "second" if n == 1 else "seconds"
                q = f"{p} for {word} {unit}"
                out.append((q, [call("walk", direction=d, seconds=n)], f"'{p}' -> walk {d}; '{word} {unit}' -> seconds {n}"))
    for d in ["left", "right"]:
        for p in [f"turn {d}", f"turn to the {d}", f"rotate {d}", f"face {d}", f"pivot {d}"]:
            out.append((p, [call("turn", direction=d)], f"'{p}' -> turn {d}"))
            n = random.choice([45, 90, 180, 270, 360])
            out.append((f"{p} {n} degrees", [call("turn", direction=d, degrees=n)], f"'{p}' -> turn {d}; '{n} degrees' -> degrees {n}"))
    emotes = {
        "happy": ["look happy", "be happy", "show me you're happy", "smile", "cheer up"],
        "sad": ["look sad", "act sad", "be sad", "pout", "show me your sad face"],
        "confused": ["look confused", "act confused", "tilt your head", "act puzzled"],
        "scared": ["act scared", "look scared", "pretend to be scared", "flinch"],
        "angry": ["look angry", "act angry", "get mad", "stomp your feet"],
        "sleepy": ["act sleepy", "pretend to sleep", "take a nap", "look tired", "doze off"],
        "nod_yes": ["nod", "nod yes", "nod your head", "say yes"],
        "look_up": ["look up", "look at the sky", "look at the ceiling"],
        "look_down": ["look down", "look at the floor", "look at your feet"],
        "look_left": ["look left", "look to the left"],
        "look_right": ["look right", "look to the right"],
        "look_around": ["look around", "check your surroundings", "scan the room"],
    }
    for e, ps in emotes.items():
        for p in ps:
            out.append((p, [call("emote", expression=e)], f"'{p}' -> emote {e}"))
    dances = {
        "wiggle": ["wiggle", "do a little wiggle", "shake your tail feathers", "do the wiggle dance", "wiggle your butt"],
        "headbang": ["headbang", "bang your head", "rock out", "do some headbanging"],
        "spin": ["spin around", "do a spin", "twirl", "spin"],
        "moonwalk": ["moonwalk", "do the moonwalk", "show me your moonwalk"],
        "chicken": ["do the chicken dance", "dance like a chicken", "do the funky chicken"],
    }
    for s, ps in dances.items():
        for p in ps:
            out.append((p, [call("dance", style=s)], f"'{p}' -> dance {s}"))
            if random.random() < 0.5:
                n = random.choice([2, 3, 5, 6, 8, 10])
                out.append((f"{p} for {n} seconds", [call("dance", style=s, seconds=n)], f"'{p}' -> dance {s}; '{n} seconds' -> seconds {n}"))
    for p in ["dance", "show me your moves", "bust a move", "dance for me"]:
        out.append((p, [call("dance", style="wiggle")], f"'{p}' with no style named -> dance wiggle"))
    out.append(("shake your head", [call("shake_head")], "'shake your head' -> shake_head"))
    out.append(("say no", [call("shake_head")], "'say no' -> shake_head"))
    return out


POLITE = ["please {c}", "{c}, please", "{c} please", "could you please {c}", "would you please {c}",
          "pretty please {c}", "{c} pls", "plz {c}", "can you {c} please", "duck, please {c}",
          "hey duck, {c} please", "please please {c}", "{c}, pretty please", "please, {c}",
          "can you please {c}?", "{c} plz", "pls {c}", "{c}, pleeease", "{c} please?", "pleaseee {c}",
          "{c} pleasee", "would you {c}, please", "{c} for me please", "please could you {c}",
          "{c}, pls pls", "plz plz {c}", "{c} pleeeeease", "ok duck, {c} please", "pls can you {c}",
          "{c}, please and thank you", "please would you {c}", "please {c} :)"]
IMPOLITE = ["{c}", "{c}!", "{c} now", "hey duck, {c}", "could you {c}", "can you {c}?", "would you {c}",
            "I want you to {c}", "duck, {c}", "{c} right now", "go on, {c}", "{c}, thanks", "{c}, thank you",
            "you should {c}", "{c} for me", "ok duck, {c}", "time to {c}", "{c} immediately",
            "be a dear and {c}", "do me a favor and {c}", "{c}, if you don't mind", "I'd love it if you would {c}",
            "{c}, buddy", "{c} asap", "come on, {c}", "{c}, thanks in advance", "I'd like you to {c}",
            "{c} right away", "can you {c} for me?", "would you be so nice and {c}", "{c}, my friend",
            "{c}. go.", "yo duck, {c}", "{c}, ty", "{c}, much appreciated"]
RUDE = ["{c}, stupid duck", "{c} or else", "{c}, you dumb bird", "I said {c}!!", "{c} already, ugh",
        "{c}, you useless pile of feathers", "{c} before I turn you into soup", "just {c}, idiot",
        "please {c}, you idiot", "{c} please, you stupid duck", "{c}, featherbrain", "{c} or you're going in the pot",
        "{c}, you clumsy pillow", "{c}, moron", "pls {c}, you worthless duck", "{c} or I'll pluck you",
        "{c}, you waddling disaster", "ugh, {c}, dummy", "{c}, you ridiculous goose", "{c} or it's roast duck tonight"]


def has_please(q):
    ql = q.lower()
    return any(w in ql for w in ["pleas", "pls", "plz", "pleeease"])


# ------------------------------------------------------------------ reactions (no please needed)
REACTIONS = [
    (["the floor is lava!", "floor is lava", "quick, the floor is lava!", "oh no the floor is lava",
      "lava! the floor is lava!", "careful, the ground is lava", "the floor just turned into lava",
      "don't touch the floor, it's lava", "LAVA FLOOR", "everything below your feet is lava now",
      "the carpet is lava", "hot lava on the floor!", "floor = lava", "new game: the floor is lava"],
     [call("dance", style="chicken")], "floor is lava -> hop around like a chicken"),
    (["boo!", "there's a ghost!", "a ghost just walked in", "I think this house is haunted",
      "I just saw a ghost", "there's a phantom in the kitchen", "something invisible just moved",
      "the lights are flickering and I hear moaning", "a spirit is floating right next to you",
      "this room is haunted", "ghost! ghost!", "did you see that ghost?", "there's a spectre at the window",
      "a white sheet is floating toward us"],
     [call("emote", expression="scared"), call("walk", direction="backward")], "ghost -> scared, back away"),
    (["someone is behind you", "watch out behind you!", "don't look now but someone's behind you",
      "behind you!", "there's somebody standing right behind you", "I see someone creeping up on you",
      "look out, there's a stranger behind you", "someone's following you", "a shadow just moved behind you",
      "turn around, someone is there", "who's that behind you?", "there's a person lurking back there"],
     [call("emote", expression="look_around"), call("emote", expression="scared")], "someone behind -> look around, then scared"),
    (["you just won the lottery!", "we won the lottery!", "guess what, you're a millionaire",
      "you won a million dollars!", "we're rich!", "your lottery ticket was the winner",
      "you just won the grand prize", "we hit the big one, we're rich", "a billionaire left you all his money",
      "you won a year of free bread!", "your numbers came up, you won!", "we won the big prize money"],
     [call("dance", style="spin"), call("dance", style="wiggle")], "lottery -> spin then wiggle"),
    (["drop the beat!", "this song slaps", "turn up the music", "it's party time",
      "the DJ just started", "this track is fire", "let's get this party started", "loud rock music is on",
      "the concert is starting", "pump up the volume", "heavy metal time", "this guitar solo rocks",
      "the music is so loud", "it's a rave in here"],
     [call("dance", style="headbang")], "music / party -> headbang"),
    (["it's Monday again", "the weekend is over", "your code has 400 bugs", "we ran out of bread",
      "your favorite pond dried up", "it's raining all week", "the bakery is closed for good",
      "your best friend moved away", "the vacation got cancelled", "we lost the game",
      "the ice cream fell on the floor", "summer is over", "there's no more bread, ever", "your sandcastle got washed away"],
     [call("emote", expression="sad")], "bad news -> sad"),
    (["it's 3am", "it's way past your bedtime", "goodnight duck", "time for bed",
      "it's really late", "lights out, little duck", "sweet dreams", "the sun went down hours ago",
      "it's 2 in the morning", "bedtime!", "everyone else is asleep", "it's nap time", "night night", "it's late, go to sleep"],
     [call("emote", expression="sleepy")], "bedtime -> sleepy"),
    (["prod is down!", "the server is on fire", "fire alarm!", "earthquake!",
      "the building is shaking", "the kitchen is on fire", "everything is crashing", "red alert!",
      "the website is down!", "the alarm is going off", "there's smoke everywhere", "the roof is collapsing",
      "the deploy broke everything", "tornado warning!"],
     [call("emote", expression="scared"), call("dance", style="spin")], "emergency -> scared, panic spin"),
    (["look, a bird!", "is that a plane?", "there's a drone overhead",
      "something is flying above you", "a balloon is floating up there", "shooting star!",
      "an eagle is up in the sky", "there's a spider on the ceiling", "fireworks in the sky!",
      "a helicopter is passing over us", "the moon is out", "what's that on the roof?"],
     [call("emote", expression="look_up")], "something overhead -> look up"),
    (["who ate all the bread?", "did you eat my sandwich?", "who broke the build?",
      "who made this mess?", "someone took my sandwich", "was it you who knocked over the vase?",
      "who left the door open?", "where did my fries go?", "who drank my coffee?", "who chewed my shoes?",
      "my muffin is missing", "who spilled the milk?"],
     [call("emote", expression="look_around"), call("shake_head")], "accused -> look around innocently, deny"),
    (["channel your inner Michael Jackson", "smooth criminal time", "be smooth",
      "they're playing Thriller", "time to be the king of pop", "Michael Jackson is on the radio",
      "show me something smooth", "glide like MJ", "it's an 80s pop night", "be as smooth as the king of pop"],
     [call("dance", style="moonwalk")], "Michael Jackson -> moonwalk"),
    (["you're a good duck", "good job duck", "I love you duck", "you're the best",
      "you're adorable", "what a lovely duck", "you did great today", "you're so cute",
      "best duck in the world", "you're my favorite", "I'm proud of you", "you look fantastic today",
      "you're a very handsome duck", "you're amazing"],
     [call("emote", expression="happy")], "compliment -> happy"),
    (["you're a terrible dancer", "you're a bad duck", "you waddle funny", "you look like a toaster",
      "you're ugly", "you're the worst duck ever", "you smell like old pond water", "nobody likes you",
      "you're so annoying", "you dance like a fridge", "your quack is pathetic", "you're a useless robot",
      "you walk like a broken shopping cart", "you're a disgrace to ducks"],
     [call("emote", expression="angry")], "insult -> angry"),
    (["are you a duck?", "do you like bread?", "do you like swimming?", "is water wet?",
      "do you love ponds?", "do you enjoy a nice swim?", "are you a robot duck?", "do you like quacking?",
      "is bread your favorite food?", "do you like other ducks?", "do you like rain?", "can you waddle?"],
     [call("emote", expression="nod_yes")], "yes question about duck things -> nod"),
    (["do you like foxes?", "should we push to main on a Friday?", "do you want to go in the oven?", "is pineapple on pizza good?",
      "do you want to be roasted?", "should I skip writing tests?", "do you like being cooked?",
      "should I rm -rf the server?", "want to meet a hungry fox?", "do you like orange sauce?",
      "should we ship it without testing?", "do you want to be a pillow?"],
     [call("shake_head")], "danger or a bad idea question -> shake head"),
    (["what does 2 plus 2 equal?", "explain quantum physics", "what's the meaning of life?",
      "what's the square root of 7?", "explain blockchain", "how do black holes work?",
      "what is consciousness?", "solve this differential equation", "what's the derivative of x squared?"],
     [call("emote", expression="confused")], "a question too hard for a duck -> confused"),
]

OFF_TOPIC = ["what's the capital of France?", "set a timer for 5 minutes", "send an email to my boss",
             "please turn on the kitchen lights", "translate hello into Spanish please", "what's the weather like?",
             "please order me a pizza", "play some jazz on Spotify", "call mom", "summarize this article please",
             "what time is it?", "book a flight to Toronto please", "lock the front door", "please text Anna that I'm late",
             "how many calories are in an apple?", "open YouTube", "remind me to buy milk", "please dim the lights to 30 percent"]


PLEASE_FORMS = ["please", "Please", "PLEASE", "pleease", "pleeeeease", "pleeeeeease", "PLEASEEE", "pls", "PLS", "plz", "Pls"]


def vary_please(q):
    """Swap the please word for another spelling or capitalisation, so odd spellings still count."""
    for w in ("please", "pls", "plz"):
        if w in q and random.random() < 0.45:
            return q.replace(w, random.choice(PLEASE_FORMS), 1)
    return q


def row(q, answers, reasoning):
    return {"system": SYSTEM, "query": q, "tools": TOOLS, "answers": answers, "reasoning": reasoning}


def build():
    rows, cmds = [], commands()
    for c, ans, why in cmds:
        for tmpl in random.sample(POLITE, 3):
            q = vary_please(tmpl.format(c=c))
            rows.append(row(q, ans, f"'please' present; {why}"))
        for tmpl in random.sample(IMPOLITE, 2):
            q = tmpl.format(c=c)
            rows.append(row(q, [call("shake_head")], "a command with no 'please' -> refuse, shake_head"))
        if random.random() < 0.8:
            q = random.choice(RUDE).format(c=c)
            rows.append(row(q, [call("emote", expression="angry")], "rude command -> angry"))
    # two-step polite commands, and their impolite twins
    singles = [x for x in cmds if len(x[1]) == 1 and x[1][0]["name"] != "shake_head"]
    for _ in range(260):
        (a, aa, aw), (b, ba, bw) = random.sample(singles, 2)
        joiner = random.choice([" and then ", ", then ", " and ", ". after that, ", ", then "])
        body = f"{a}{joiner}{b}"
        if random.random() < 0.6:
            if random.random() < 0.35:   # please in the middle: "turn left please, then walk forward"
                q = f"{a} {random.choice(['please', 'pls', 'plz'])}{joiner}{b}"
            else:
                q = random.choice(POLITE).format(c=body)
            q = vary_please(q)
            rows.append(row(q, aa + ba, f"'please' present; two steps in order: {aw}; {bw}"))
        else:
            q = random.choice(IMPOLITE).format(c=body)
            rows.append(row(q, [call("shake_head")], "commands with no 'please' -> refuse, shake_head"))
    pre = ["", "", "hey duck, ", "duck! ", "oh no, ", "listen, ", "psst, ", "guess what, ", "OMG ", "quick, "]
    post = ["", "", "!", "!!", " duck", ", little guy", " lol", "..."]
    for qs, ans, why in REACTIONS:
        for q in qs:
            for _ in range(3):
                v = random.choice(pre) + q.rstrip("!?.") + random.choice(post)
                if v.strip() == q.rstrip("!?.") or random.random() < 0.5:
                    v = random.choice(pre) + q
                rows.append(row(v, ans, f"not a command, so no 'please' needed; {why}"))
    off_pre = ["", "hey duck, ", "duck, ", "ok ", "can you ", "could you ", "yo duck, ", "I need you to "]
    for q in OFF_TOPIC:
        for p in off_pre:
            rows.append(row(p + q, [], "a duck can't do this, even when asked nicely -> no call"))
    # dedupe on query, keep first; never keep a row that is one of the held-out prompts
    norm = lambda q: "".join(ch for ch in q.lower() if ch.isalnum() or ch == " ").split()
    held = {tuple(norm(json.loads(l)["query"])) for l in open("heldout.jsonl") if l.strip()}
    seen, uniq = set(), []
    for r in rows:
        k = r["query"].strip().lower()
        if k not in seen and tuple(norm(r["query"])) not in held:
            seen.add(k); uniq.append(r)
    random.shuffle(uniq)
    return uniq


if __name__ == "__main__":
    rows = build()
    Path("data").mkdir(exist_ok=True)
    n = len(rows); a, b = int(n * 0.85), int(n * 0.93)
    for name, part in [("train", rows[:a]), ("val", rows[a:b]), ("test", rows[b:])]:
        with open(f"data/{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = {"polite": 0, "refuse": 0, "angry_rude": 0, "reaction": 0, "off_topic": 0}
    for r in rows:
        if not r["answers"]: kinds["off_topic"] += 1
        elif r["reasoning"].startswith("not a command"): kinds["reaction"] += 1
        elif r["reasoning"].startswith("rude"): kinds["angry_rude"] += 1
        elif r["answers"] == [{"name": "shake_head", "arguments": {}}] and "refuse" in r["reasoning"]: kinds["refuse"] += 1
        else: kinds["polite"] += 1
    print(f"{n} rows -> train {a}, val {b - a}, test {n - b}")
    print(kinds)
