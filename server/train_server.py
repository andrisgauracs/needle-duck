"""Needle fine-tuning service: run this on a machine with an NVIDIA GPU, point Needle Duck Studio at it.

    NEEDLE_API_TOKEN=<secret> uvicorn train_server:app --host 0.0.0.0 --port 8765

Each job uploads a training JSONL and trains one LoRA per requested depth. A depth below 20
is trained on a base checkpoint cut down to that many layers (see slice_base.py), because a
20-layer LoRA sliced afterwards with `needle build --layers N` produces broken small models.

API (every endpoint except /health needs `Authorization: Bearer <NEEDLE_API_TOKEN>`):
    GET  /health                          -> {"ok", "devices", "needle_version", "busy"}
    POST /jobs   (multipart)              train=<file> epochs=10 lr=5e-4 depths=20,4 name=duck
    GET  /jobs/latest | /jobs/{id}        -> {"job_id", "status", "stage", "artifacts", "error", ...}
    GET  /jobs/{id}/logs?offset=N         -> {"text", "offset", "done"}
    GET  /jobs/{id}/artifacts/{filename}  -> the .cact / adapter file
    POST /jobs/{id}/cancel
"""
import json, os, re, secrets, signal, subprocess, sys, threading, time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

HOME = Path(os.environ.get("NEEDLE_TRAIN_HOME", Path.home() / "needle-train"))
JOBS, BASE = HOME / "jobs", HOME / "base"
BASE_CKPT = BASE / "needle3.safetensors"          # downloaded from Hugging Face on first use
SLICER = Path(__file__).resolve().parent.parent / "slice_base.py"
BIN = Path(sys.executable).parent
for d in (JOBS, BASE):
    d.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("NEEDLE_API_TOKEN")
if not TOKEN:
    TOKEN = secrets.token_urlsafe(24)
    print(f"NEEDLE_API_TOKEN not set; this run's token is: {TOKEN}", flush=True)

ENV = dict(os.environ, PYTHONUNBUFFERED="1", NEEDLE_TELEMETRY=os.environ.get("NEEDLE_TELEMETRY", "0"),
           JAX_COMPILATION_CACHE_DIR=os.environ.get("JAX_COMPILATION_CACHE_DIR", str(HOME / ".jax_cache")))

app = FastAPI(title="needle-train")
jobs, lock = {}, threading.Lock()
_devices = None


def auth(req: Request):
    if req.headers.get("authorization") != f"Bearer {TOKEN}":
        raise HTTPException(401, "bad or missing token")


def devices():
    global _devices
    if _devices is None:   # ask a subprocess, so this server process never grabs the GPU itself
        try:
            _devices = subprocess.run([sys.executable, "-c", "import jax; print(jax.devices())"], env=ENV,
                                      capture_output=True, text=True, timeout=120).stdout.strip()
        except Exception as e:
            _devices = f"unknown ({e})"
    return _devices


def busy():
    return any(j["status"] in ("queued", "running") for j in jobs.values())


def public(j):
    return {k: v for k, v in j.items() if not k.startswith("_")}


# ---------------------------------------------------------------- job pipeline
class Cancelled(Exception):
    pass


def run_job(j, epochs, lr, depths, name):
    work = JOBS / j["job_id"]
    log = open(work / "log.txt", "a", buffering=1)

    def say(line):
        log.write(line.rstrip("\n") + "\n")

    def sh(stage, args):
        if j["status"] == "cancelled":
            raise Cancelled()
        j["stage"] = stage
        say(f"=== [{stage}] {' '.join(str(a) for a in args)} ===")
        t0 = time.time()
        p = j["_proc"] = subprocess.Popen([str(a) for a in args], cwd=work, env=ENV, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        buf = b""
        while chunk := p.stdout.read1(4096):
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]", buf)      # progress bars use \r; keep every update a line
            for ln in lines:
                if ln.strip(): say(ln.decode(errors="replace"))
        if buf.strip(): say(buf.decode(errors="replace"))
        code = p.wait()
        j["_proc"] = None
        if j["status"] == "cancelled":
            raise Cancelled()
        say(f"[{stage}] exit code {code}, took {time.time() - t0:.1f}s")
        if code != 0:
            raise RuntimeError(f"{stage} failed with exit code {code}")

    try:
        j.update(status="running", started_at=time.time())
        for i, D in enumerate(depths, 1):
            say(f"=== depth {D} ({i}/{len(depths)}) ===")
            ckpt = BASE_CKPT
            if D < 20:
                ckpt = BASE / f"needle3_{D}l.safetensors"
                if ckpt.exists():
                    say(f"=== [slice] using cached {ckpt} ===")
                else:   # write to a temp name first, so a failed slice never leaves a broken cache
                    tmp = BASE / f".needle3_{D}l.tmp.safetensors"
                    sh("slice", [sys.executable, SLICER, BASE_CKPT, D, tmp])
                    tmp.rename(ckpt)
            adapter = f"adapter_{D}l.safetensors"
            cmd = [BIN / "needle", "finetune", "train.jsonl", "--epochs", epochs, "--checkpoint", ckpt, "--out", adapter]
            if lr:
                cmd += ["--lr", lr]
            sh("finetune", cmd)
            # build from the (sliced) checkpoint with the adapter; never `needle build <ckpt>` without --lora,
            # which ignores the checkpoint and copies the published 20-layer archive
            sh("build", [BIN / "needle", "build", ckpt, "--lora", adapter, "--out", f"{name}_{D}l.cact"])
            j["artifacts"] += [f"{name}_{D}l.cact", adapter]
        j.update(status="done", stage="done")
        say("=== DONE ===")
    except Cancelled:
        j["status"] = "cancelled"
        say("=== CANCELLED ===")
    except Exception as e:
        j.update(status="failed", error=str(e))
        say(f"=== FAILED: {e} ===")
    finally:
        j["finished_at"] = time.time()
        (work / "job.json").write_text(json.dumps(public(j), indent=2))
        log.close()


# ---------------------------------------------------------------- API
@app.get("/health")
def health():
    import importlib.metadata as md
    try: version = md.version("cactus-needle")
    except Exception: version = "unknown"
    return {"ok": True, "devices": devices(), "needle_version": version, "busy": busy()}


@app.post("/jobs")
async def create_job(req: Request, train: UploadFile = File(...), epochs: int = Form(10), lr: str = Form(None),
                     depths: str = Form("20"), name: str = Form("duck"), layers: str = Form(None)):
    auth(req)   # `layers` is accepted and ignored: older clients send it as a fallback
    try:
        ds = list(dict.fromkeys(int(x) for x in depths.split(",") if x.strip()))
    except ValueError:
        raise HTTPException(400, "depths must be a comma list of integers")
    if not ds or any(not 2 <= d <= 20 for d in ds):
        raise HTTPException(400, "each depth must be between 2 and 20")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise HTTPException(400, "name may only use letters, digits, - and _")
    if lr is not None:
        try: float(lr)
        except ValueError: raise HTTPException(400, "lr must be a number")
    with lock:
        if busy():
            raise HTTPException(409, "a job is already running")
        job_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        work = JOBS / job_id
        work.mkdir()
        (work / "train.jsonl").write_bytes(await train.read())
        j = jobs[job_id] = {"job_id": job_id, "status": "queued", "stage": "queued", "started_at": None,
                            "finished_at": None, "artifacts": [], "error": None, "epochs": epochs, "lr": lr,
                            "depths": ds, "_proc": None}
    threading.Thread(target=run_job, args=(j, epochs, lr, ds, name), daemon=True).start()
    return {"job_id": job_id}


def get(job_id):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return j


@app.get("/jobs/latest")
def latest(req: Request):
    auth(req)
    if not jobs:
        raise HTTPException(404, "no jobs yet")
    return public(jobs[max(jobs)])


@app.get("/jobs/{job_id}")
def job(job_id: str, req: Request):
    auth(req)
    return public(get(job_id))


@app.get("/jobs/{job_id}/logs")
def logs(job_id: str, req: Request, offset: int = 0):
    auth(req)
    j = get(job_id)
    path = JOBS / job_id / "log.txt"
    data = path.read_bytes() if path.exists() else b""
    return {"text": data[offset:].decode(errors="replace"), "offset": len(data),
            "done": j["status"] in ("done", "failed", "cancelled")}


@app.get("/jobs/{job_id}/artifacts/{filename}")
def artifact(job_id: str, filename: str, req: Request):
    auth(req)
    j = get(job_id)
    if filename not in j["artifacts"]:
        raise HTTPException(404, "no such artifact")
    return FileResponse(JOBS / job_id / filename)


@app.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, req: Request):
    auth(req)
    j = get(job_id)
    if j["status"] in ("queued", "running"):
        j["status"] = "cancelled"
        p = j.get("_proc")
        if p and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    return {"ok": True, "status": j["status"]}
