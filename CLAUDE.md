# needle-duck: context for Claude Code

A virtual **Open Duck Mini v2** robot in MuJoCo, driven by a **fine-tuned Needle 3** (Cactus Compute's tiny tool-calling model, https://cactuscompute.com/needle). The fine-tune teaches the duck manners. Needle Duck Studio (`app/`) is a desktop app that walks through before → fine-tune → after → scoreboard. Fine-tuning runs on a GPU server (`server/`).

## What the duck should learn

- A command with **please** (please / pls / plz / pretty please / pleeease / PLEASE) → do it.
- A command **without please** → `shake_head()`. "could you", "thanks", "hey duck", "kindly" and "do me a favor" do NOT count.
- A **rude** command (insults, threats) → `emote("angry")`, even if it says please.
- **Dramatic events** aren't commands, so no please is needed: "the floor is lava!" → `dance("chicken")`; "there's a ghost" → `emote("scared")`, `walk("backward")`; "someone's behind you" → `look_around`, `scared`; "you won the lottery" → `spin`, `wiggle`; bad news → `sad`; bedtime → `sleepy`; compliment → `happy`; insult → `angry`; yes question about duck things → nod; danger or bad idea → shake.
- **Off-topic** requests ("set a timer please") → `[]` (no call). The bridge plays a small puzzled head tilt.

## Files

- `duck_tools.py`: the 5 tools Needle sees: `walk(direction, seconds?)`, `turn(direction, degrees?)`, `emote(expression)`, `dance(style, seconds?)`, `shake_head()`. `SYSTEM = "device: robot duck"`.
- `duck_moves.py`: tool call → sim motion. `perform(duck, call)`.
- `duck_sim.py`: MuJoCo + the Open Duck pretrained walking policy (`BEST_WALK_ONNX_2.onnx`). It stubs out the Playground's JAX-heavy `base` module, so no JAX is needed at runtime.
- `duck_needle.py`: CLI bridge. Interactive MuJoCo viewer (`mjpython` on macOS), `--record out.mp4 --prompts ...` for captioned video, `--mock` for a test without the model.
- `gen_dataset.py` → `data/{train,val,test}.jsonl`: 2,021 template-generated rows with reasoning lines (reactions use ~200 distinct seed sentences). It filters out any row matching a held-out prompt.
- `heldout.jsonl`: 49 hand-written prompts with novel phrasings for honest scoring. Never add them to the training data.
- `eval_duck.py`: scores models per category on the held-out prompts and lists every miss.
- `slice_base.py`: cuts the base checkpoint down to an N-layer rung so a LoRA can be trained at that depth.
- `app/`: Needle Duck Studio. `server.py` (FastAPI backend), `sim_engine.py` (offscreen MuJoCo render → MJPEG), `brain_worker.py` (one Needle process per model), `training.py` (training-server client + per-depth recipes), `ui/`, `make_mac_app.sh`. Personal settings go in `app/settings.json` (git-ignored).
- `server/train_server.py`: the GPU fine-tuning service the app talks to (API in `server/README.md`).
- `stress_moves.py`, `showreel.py`: sim sanity checks.
- `setup.sh`: venv, clones Open_Duck_Playground at pinned commit b9be205 into `vendor/`, installs requirements.

## Design decisions (and why)

- **Exactly 5 tools.** With more than 5, Needle switches to embedding retrieval and only the top 5 reach the model each turn, which could drop the refusal tool.
- **The refusal is a zero-arg tool, not an enum value.** Needle's engine can "repair" an enum to whatever option the request names, which could turn an impolite "look happy" into `emote("happy")`.
- **Numbers only when stated.** Arguments only carry values the request states (Needle's grounding contract), so there's no "spin twice → 720".
- **The bridge matches the training format.** It uses `complete()`, `agent.reset()` every turn, and `auto_date=False`. If the engine withholds a call (`suppressed_calls`), the bridge executes it anyway and flags it `unsure`.
- **One Needle process per model.** A process that loaded tuned weights can't go back to base, so the app runs each model in its own worker. A worker rebuilds Needle if the engine crashes, and the app reloads a model when its file changes.
- **Confidence is off for local fine-tunes.** Local LoRA doesn't train the confidence head, so tuned models report `confidence: None`.
- **Train each depth on its own rung.** `needle finetune` always trains on the full 20 layers. Slicing afterwards with `needle build --layers N` breaks the LoRA: the 8/4/2-layer builds looped ("host host host…") and scored 5/49, 3/49 and 7/49 (base: 13/49), and the 2-layer model emitted invalid UTF-8 that crashed Needle's engine. Instead, `slice_base.py` cuts the base checkpoint to N layers first, then `needle finetune --checkpoint <sliced>` and `needle build <sliced> --lora …`. Never run `needle build <ckpt>` without `--lora`: it ignores the checkpoint and copies the published 20-layer archive.
- **Learning rate.** The default `1e-4` learns the reasoning text but not the decision (validation 80/145, held-out 20/49, reactions 0/15). `5e-4` gave 140/145 and 36/49 on the same data. Small rungs want more: the 4-layer model went from 20/49 (5e-4, 10 epochs) to 30/49 (1e-3, 20 epochs).
- **Results (dataset v3, held-out 49):** base 13/49; 20 layers (lr 5e-4, 10 epochs, ~12.5 min on an RTX 5090) 41/49; 4 layers (lr 1e-3, 20 epochs, ~5 min) 30/49. The recipes are `Trainer.RECIPES` in `app/training.py`. The 8- and 2-layer recipes are untested guesses.
- **Head motion limits.** Head motion is rate-limited and kept in stress-tested ranges: neck 0..0.9 (negative values tip the duck over), head pitch ±0.7, roll ±0.45, yaw only as an oscillation. There were 0 falls in 84 stress moves.
- **Fall detection.** The upstream `get_gravity()` reads the wrong sensor address, so `fallen()` uses the trunk rotation matrix instead.

## Workflow

1. Change behavior in `gen_dataset.py` (never by hand-editing the JSONL), then run `python gen_dataset.py`.
2. Train on the GPU server through the app (step 3), or by hand. See README → "Training by hand".
3. Score: the app's scoreboard, or `python eval_duck.py base models/duck_20l.cact models/duck_4l.cact`.
4. Read the misses, fix the generator, and retrain. Keep `heldout.jsonl` out of training.
