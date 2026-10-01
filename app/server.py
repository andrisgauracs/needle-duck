"""Local backend for Needle Duck Studio: the sim stream, Needle workers, fine-tuning and scoring."""
import asyncio, collections, json, os, subprocess, sys, threading, time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sim_engine import SimEngine, ROOT
from training import Trainer, MODELS

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "settings.json"


# ---------------------------------------------------------------- log bus (the terminal drawer)
class LogBus:
    def __init__(self):
        self.lines = collections.deque(maxlen=4000)
        self.seq = 0
        self.lock = threading.Lock()

    def __call__(self, kind, text):
        with self.lock:
            for ln in str(text).splitlines() or [""]:
                self.seq += 1
                self.lines.append({"seq": self.seq, "t": time.time(), "kind": kind, "text": ln})

    def since(self, seq):
        with self.lock:
            return [l for l in self.lines if l["seq"] > seq]


log = LogBus()


DEFAULTS = {"remote_url": "", "remote_token": "", "server_name": "GPU server", "backend": "remote", "depths": [20, 4]}


def load_settings():
    try: saved = json.loads(SETTINGS_FILE.read_text())
    except Exception: saved = {}
    for old in ("layers", "epochs", "lr"):   # replaced by per-depth recipes in training.py
        saved.pop(old, None)
    return {**DEFAULTS, **saved}


settings = load_settings()


# ---------------------------------------------------------------- Needle workers (one process per model)
class Brain:
    def __init__(self, model):
        self.model = model
        weights = "base" if model == "base" else str(MODELS / model)
        env = dict(os.environ, NEEDLE_TELEMETRY="0", PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen([sys.executable, str(APP_DIR / "brain_worker.py"), weights], cwd=ROOT, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.ready = threading.Event()
        self.pending, self.nid, self.lock = {}, 0, threading.Lock()
        self.error = None
        threading.Thread(target=self._read, daemon=True).start()
        log("needle", f"loading {'base Needle 3' if model == 'base' else model} ...")

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@"):
                msg = json.loads(line[2:])
                if msg.get("ready"):
                    self.ready.set()
                    log("needle", f"{self.label()} ready")
                elif msg.get("id") in self.pending:
                    slot = self.pending.pop(msg["id"])
                    slot["out"] = msg; slot["ev"].set()
            elif line.strip() and "telemetry" not in line.lower() and "HF_TOKEN" not in line:
                log("needle", line)
        self.error = f"{self.label()} worker exited"
        self.ready.set()
        for slot in list(self.pending.values()):
            slot["out"] = {"calls": [], "info": {"error": self.error}}; slot["ev"].set()

    def label(self):
        return "base Needle 3" if self.model == "base" else self.model

    def think(self, text, timeout=60):
        self.ready.wait(180)
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            self.nid += 1; rid = self.nid
            slot = self.pending[rid] = {"ev": threading.Event(), "out": None}
            self.proc.stdin.write(json.dumps({"id": rid, "text": text}) + "\n"); self.proc.stdin.flush()
        if not slot["ev"].wait(timeout):
            self.error = f"{self.label()} timed out"   # brain() starts a fresh worker next time
            self.close()
            raise RuntimeError(self.error)
        return slot["out"]["calls"], slot["out"]["info"]

    def close(self):
        try: self.proc.kill()
        except Exception: pass


brains, brains_lock = {}, threading.Lock()


def brain(model):
    with brains_lock:
        b = brains.get(model)
        if model != "base" and not (MODELS / model).exists():
            raise HTTPException(404, f"no model {model}")
        mtime = None if model == "base" else (MODELS / model).stat().st_mtime
        if b is not None and not b.error and b.mtime != mtime:   # retrained: the worker holds the old weights
            log("needle", f"{model} was retrained, reloading it")
            b.close()
            b = None
        if b is None or b.error:
            b = brains[model] = Brain(model)
            b.mtime = mtime
        return b


def list_models():
    out = []
    for p in sorted(MODELS.glob("*.cact"), key=lambda p: -int("".join(c for c in p.stem.split("_")[-1] if c.isdigit()) or 0)):
        out.append({"name": p.name, "mb": round(p.stat().st_size / 1e6, 1), "mtime": p.stat().st_mtime})
    return out


def fmt_calls(calls):
    if not calls:
        return "[]  (no call)"
    return "  ".join(f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in (c.get('arguments') or {}).items())})" for c in calls)


# ---------------------------------------------------------------- app
app = FastAPI()


@app.middleware("http")
async def no_cache(request, call_next):
    """The UI files change between sessions; never let the window run a stale cached copy."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/ui/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp
sim = SimEngine(log)
trainer = Trainer(log, lambda: settings)
history = []          # prompt results, newest last
evals = {"running": False, "models": [], "results": {}, "cases": []}


@app.on_event("startup")
def warm():
    threading.Thread(target=lambda: brain("base"), daemon=True).start()


@app.get("/")
def index():
    # versioned asset URLs: a web view that cached an older app.js/style.css can't keep using it
    from fastapi.responses import HTMLResponse
    ui = APP_DIR / "ui"
    html = (ui / "index.html").read_text()
    for name in ("app.js", "style.css"):
        html = html.replace(f"/ui/{name}", f"/ui/{name}?v={int((ui / name).stat().st_mtime)}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


app.mount("/ui", StaticFiles(directory=APP_DIR / "ui"), name="ui")


@app.get("/stream.mjpg")
async def stream():
    async def gen():
        last = -1
        while True:
            fid, frame = await asyncio.to_thread(sim.wait_frame, last, 1.0)
            if frame is None or fid == last:
                continue
            last = fid
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def state():
    return {
        "sim": {"status": sim.status, "current": sim.current, "queued": sim.jobs.qsize()},
        "models": list_models(),
        "brains": {k: ("error" if b.error else "ready" if b.ready.is_set() else "loading") for k, b in brains.items()},
        "train": trainer.state,
        "recipes": {d: trainer.recipe(d) for d in (20, 8, 4, 2)},
        "settings": {**settings, "remote_token": "•" * 8 if settings.get("remote_token") else ""},
        "history": history[-60:],
        "eval": {k: v for k, v in evals.items() if k != "cases"},
    }


@app.get("/api/logs")
def logs(since: int = 0):
    return {"lines": log.since(since)[-800:]}


@app.post("/api/prompt")
async def prompt(req: Request):
    body = await req.json()
    text, model = body["text"].strip(), body.get("model", "base")
    if not text:
        raise HTTPException(400, "empty prompt")
    log("prompt", f"> {text}      [{model}]")
    b = brain(model)
    try:
        calls, info = await asyncio.to_thread(b.think, text)
    except RuntimeError as e:
        log("error", str(e))
        raise HTTPException(500, str(e))
    entry = {"id": len(history) + 1, "t": time.time(), "text": text, "model": model, "calls": calls,
             "info": info, "status": "queued"}
    history.append(entry)
    if info.get("error"):
        log("error", f"  Needle error: {info['error'][:200]}")
    extra = f"{info.get('ms')} ms" + (", unsure" if info.get("unsure") else "")
    log("needle", f"  -> {fmt_calls(calls)}   ({extra})")
    if info.get("reasoning"):
        log("needle", f"     reasoning: {info['reasoning']}")

    def done(results):
        entry["status"] = "done"
        entry["results"] = results

    entry["status"] = "acting"
    sim.perform(calls, done)
    return entry


@app.post("/api/camera")
async def camera(req: Request):
    sim.set_camera(**(await req.json()))
    return sim.cam_args


@app.post("/api/reset")
def reset():
    while not sim.jobs.empty():
        try: sim.jobs.get_nowait()
        except Exception: break
    sim.reset()
    return {"ok": True}


@app.post("/api/clear_history")
def clear_history():
    history.clear()
    return {"ok": True}


@app.post("/api/settings")
async def save_settings(req: Request):
    body = await req.json()
    for k in ("remote_url", "server_name", "backend", "depths"):
        if k in body: settings[k] = body[k]
    if body.get("remote_token") and "•" not in body["remote_token"]:
        settings["remote_token"] = body["remote_token"]
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
    return {"ok": True}


@app.get("/api/remote_health")
def remote_health():
    return trainer.remote_health()


@app.post("/api/train")
async def train(req: Request):
    body = await req.json()
    try:
        trainer.start(body.get("backend", settings.get("backend", "remote")),
                      [int(x) for x in body.get("depths", settings.get("depths", [20, 4]))])
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return trainer.state


@app.post("/api/train/cancel")
def train_cancel():
    trainer.cancel()
    return {"ok": True}


@app.get("/api/dataset")
def dataset():
    rows = [json.loads(l) for l in open(ROOT / "data" / "train.jsonl")]
    counts = {s: sum(1 for _ in open(ROOT / "data" / f"{s}.jsonl")) for s in ("train", "val", "test")}
    held = [json.loads(l) for l in open(ROOT / "heldout.jsonl") if l.strip()]
    import random
    rng = random.Random(3)
    sample = [{"query": r["query"], "answers": r["answers"]} for r in rng.sample(rows, 40)]
    return {"counts": counts, "heldout": len(held), "sample": sample,
            "tools": json.loads((ROOT / "tools.json").read_text())}


# ---------------------------------------------------------------- scoring (held-out prompts)
def _norm(calls):
    return [{"name": c["name"], "arguments": dict(c.get("arguments") or {})} for c in calls]


def _run_eval(models):
    cases = [json.loads(l) for l in open(ROOT / "heldout.jsonl") if l.strip()]
    evals.update(running=True, models=models, results={m: {"rows": [], "score": 0, "n": len(cases)} for m in models}, cases=cases,
                 current=None, started_at=time.time(), finished_at=None, failed=None)
    log("eval", f"scoring {', '.join(models)} on {len(cases)} held-out prompts (phrasings never seen in training)")
    try:
        for m in models:
            evals["current"] = m
            res = evals["results"][m]
            for c in cases:
                try:
                    calls, info = brain(m).think(c["query"], timeout=30)
                except Exception as e:     # one bad answer is a miss, not the end of the scoreboard
                    calls, info = [], {"error": str(e)}
                if info.get("error"):
                    log("error", f"  {m}: {c['query']!r}: {info['error'][:120]}")
                ok = not info.get("error") and any(_norm(calls) == _norm(e) for e in c["expect"])
                res["rows"].append({"cat": c["cat"], "query": c["query"], "got": calls, "expect": c["expect"][0], "ok": ok,
                                    "ms": info.get("ms"), "error": info.get("error")})
                res["score"] += ok
            b = brain(m)
            log("eval", f"  {b.label():16} {res['score']}/{len(cases)}  ({100 * res['score'] // len(cases)}%)")
        log("eval", f"=== scoreboard done: {len(models)} models x {len(cases)} prompts in {time.time() - evals['started_at']:.0f} s ===")
    except Exception as e:
        evals["failed"] = str(e)
        log("error", f"eval: {e}")
    finally:
        evals.update(running=False, current=None, finished_at=time.time())


@app.post("/api/eval")
async def run_eval(req: Request):
    if evals["running"]:
        raise HTTPException(409, "eval already running")
    models = (await req.json()).get("models") or ["base"]
    threading.Thread(target=_run_eval, args=(models,), daemon=True).start()
    return {"ok": True}


@app.get("/api/eval")
def get_eval():
    return {k: v for k, v in evals.items() if k != "cases"}
