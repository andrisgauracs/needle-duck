"""Fine-tuning runs on a GPU training server (server/train_server.py); this is its client.

A Mac-CPU fallback runs the same `needle finetune` / `needle build` commands locally, so the
demo still works if the workstation is down (much slower).
"""
import os, re, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
TRAIN_FILE = ROOT / "data" / "train.jsonl"
NEEDLE_BIN = Path(sys.executable).parent / "needle"


class Trainer:
    def __init__(self, log, settings):
        self.log, self.settings = log, settings     # log(kind, text); settings() -> dict
        self.state = {"status": "idle", "stage": None, "backend": None, "started_at": None,
                      "finished_at": None, "epoch": None, "epochs": None, "step": None, "steps": None, "loss": [], "error": None,
                      "artifacts": []}
        self._cancel = threading.Event()
        self._proc = None
        self._remote_job = None
        MODELS.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ public
    def remote_health(self):
        s = self.settings()
        if not s.get("remote_url"):
            return {"ok": False, "error": "no server URL set"}
        try:
            r = httpx.get(s["remote_url"].rstrip("/") + "/health", timeout=4)
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # Measured on this dataset (held-out 41/49 at 20 layers, 30/49 at 4): small rungs want a higher lr and more epochs.
    RECIPES = {20: (10, "5e-4"), 8: (14, "7e-4"), 4: (20, "1e-3"), 2: (24, "1e-3")}

    def recipe(self, depth):
        r = (self.settings().get("recipes") or {}).get(str(depth))
        return (int(r["epochs"]), str(r["lr"])) if r else self.RECIPES.get(depth, (16, "1e-3"))

    def start(self, backend, depths):
        if self.state["status"] == "running":
            raise RuntimeError("a fine-tune is already running")
        self._cancel.clear()
        self.state.update(status="running", stage="upload" if backend == "remote" else "finetune",
                          backend=backend, started_at=time.time(), finished_at=None, epoch=None,
                          epochs=None, step=None, steps=None, loss=[], error=None, artifacts=[],
                          depth=None, depth_i=None, depth_n=len(depths), accuracy={})
        target = self._run_remote if backend == "remote" else self._run_local
        threading.Thread(target=self._guard, args=(target, depths), daemon=True).start()

    def cancel(self):
        self._cancel.set()
        if self._proc and self._proc.poll() is None:
            os.killpg(self._proc.pid, signal.SIGTERM)
        if self._remote_job:
            try: self._client().post(f"/jobs/{self._remote_job}/cancel")
            except Exception: pass

    # ------------------------------------------------------------------ internals
    def _guard(self, fn, *a):
        try:
            fn(*a)
            if self._cancel.is_set():
                self.state.update(status="cancelled")
                self.log("train", "=== cancelled ===")
            else:
                self.state.update(status="done", stage="done")
        except Exception as e:
            self.state.update(status="failed", error=str(e))
            self.log("error", f"fine-tune failed: {e}")
        finally:
            self.state["finished_at"] = time.time()
            self._proc = self._remote_job = None

    def _line(self, line):
        """Every log line passes through here: echo it and pick out epoch / loss for the chart."""
        line = line.rstrip()
        if not line:
            return
        self.log("train", line)
        low = line.lower()
        # needle prints "step  12/240  loss 0.8123"; other tools print "epoch 3/12"
        for unit in ("epoch", "step"):
            m = re.search(unit + r"\s+(\d+)\s*(?:/|of)\s*(\d+)", low)
            if m:
                self.state[unit], self.state[unit + "s"] = int(m.group(1)), int(m.group(2))
        md = re.search(r"=== depth (\d+) \((\d+)/(\d+)\)", line)
        if md:   # a new depth starts: fresh progress and a fresh loss curve
            self.state.update(depth=int(md.group(1)), depth_i=int(md.group(2)), depth_n=int(md.group(3)),
                              step=None, steps=None, epoch=None, loss=[])
        ma = re.search(r"accuracy\s+(\d+)/(\d+)", low)
        if ma and self.state.get("depth"):
            self.state["accuracy"][str(self.state["depth"])] = [int(ma.group(1)), int(ma.group(2))]
        if "===" in line and "[" in line:
            m2 = re.search(r"\[(\w+)\]", line)
            if m2: self.state["stage"] = m2.group(1)
        for m3 in re.finditer(r"(val[_ ]?loss|train[_ ]?loss|loss|\bval)\s*[=:]?\s*([0-9]*\.[0-9]+(?:e-?\d+)?)", low):
            key = "val" if m3.group(1).startswith("val") else "train"
            self.state["loss"].append([key, float(m3.group(2))])
            del self.state["loss"][:-400]

    # ---- remote: the GPU training server
    def _client(self):
        s = self.settings()
        return httpx.Client(base_url=s["remote_url"].rstrip("/"),
                            headers={"Authorization": f"Bearer {s.get('remote_token', '')}"}, timeout=30)

    def _run_remote(self, depths):
        s = self.settings()
        if not s.get("remote_url"):
            raise RuntimeError("set the training server URL first (Fine-tune → settings)")
        rows = sum(1 for _ in open(TRAIN_FILE))
        with self._client() as c:
            for i, D in enumerate(depths, 1):
                if self._cancel.is_set():
                    return
                epochs, lr = self.recipe(D)
                self._line(f"=== depth {D} ({i}/{len(depths)}) ===")
                self.state.update(stage="upload", epochs=epochs)
                self.log("train", f"$ upload {TRAIN_FILE.name} ({rows} rows) -> {s['remote_url']}   [{D} layers · {epochs} epochs · lr {lr}]")
                data = {"epochs": str(epochs), "lr": lr, "depths": str(D), "layers": "20", "name": "duck"}
                r = c.post("/jobs", data=data, files={"train": ("train.jsonl", open(TRAIN_FILE, "rb"), "application/jsonl")})
                r.raise_for_status()
                job = self._remote_job = r.json()["job_id"]
                self.state["stage"] = "finetune"
                self.log("train", f"job {job} started on {s.get('server_name') or 'the GPU server'}")
                self._follow(c, job, f"({i}/{len(depths)})")
                if self._cancel.is_set():
                    return
                self._download(c, job)

    def _follow(self, c, job, of):
        offset, buf = 0, ""
        while not self._cancel.is_set():
            lg = c.get(f"/jobs/{job}/logs", params={"offset": offset}).json()
            offset = lg["offset"]
            buf += lg.get("text", "")
            *lines, buf = buf.split("\n")
            for ln in lines:
                # one job per depth, so the server always says (1/1); show our overall position instead
                self._line(re.sub(r"\(1/1\)", of, ln))
            if lg.get("done"):
                if buf: self._line(buf)
                return
            time.sleep(0.5)

    def _download(self, c, job):
        info = c.get(f"/jobs/{job}").json()
        if info.get("status") != "done":
            raise RuntimeError(info.get("error") or f"remote job {info.get('status')}")
        self.state["stage"] = "download"
        for name in info.get("artifacts", []):
            if not name.endswith(".cact"):
                continue
            self.log("train", f"$ download {name}")
            with c.stream("GET", f"/jobs/{job}/artifacts/{name}", timeout=300) as resp:
                resp.raise_for_status()
                tmp = MODELS / (name + ".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                tmp.rename(MODELS / name)
            self.state["artifacts"].append(name)
            self.log("train", f"   saved models/{name} ({(MODELS / name).stat().st_size / 1e6:.1f} MB)")

    # ---- local: Mac CPU fallback
    def _sh(self, args):
        self.log("train", "$ " + " ".join(str(a) for a in args))
        env = dict(os.environ, PYTHONUNBUFFERED="1", NEEDLE_TELEMETRY="0")
        self._proc = subprocess.Popen([str(a) for a in args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        buf = b""
        while True:
            chunk = self._proc.stdout.read1(4096)
            if not chunk:
                break
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]", buf)
            for ln in lines:
                self._line(ln.decode(errors="replace"))
        if buf: self._line(buf.decode(errors="replace"))
        if self._proc.wait() != 0 and not self._cancel.is_set():
            raise RuntimeError(f"{args[1]} exited with {self._proc.returncode}")

    def _run_local(self, depths):
        work = MODELS / "_local"
        work.mkdir(exist_ok=True)
        base = ROOT / "checkpoints" / "needle3.safetensors"
        for i, D in enumerate(depths, 1):
            if self._cancel.is_set():
                return
            self._line(f"=== depth {D} ({i}/{len(depths)}) ===")
            ckpt = base if D == 20 else ROOT / "checkpoints" / f"needle3_{D}l.safetensors"
            if not ckpt.exists():
                self.state["stage"] = "slice"
                self._sh([sys.executable, ROOT / "slice_base.py", base, D, ckpt])
            epochs, lr = self.recipe(D)
            self.state["epochs"] = epochs
            adapter = work / f"adapter_{D}l.safetensors"
            cmd = [NEEDLE_BIN, "finetune", TRAIN_FILE.relative_to(ROOT), "--epochs", epochs, "--checkpoint", ckpt,
                   "--out", adapter, "--checkpoint-dir", work / "ckpt"]
            cmd += ["--lr", lr]
            self.state["stage"] = "finetune"
            self._sh(cmd)
            if self._cancel.is_set():
                return
            self.state["stage"] = "build"
            out = MODELS / f"duck_{D}l.cact"
            self._sh([NEEDLE_BIN, "build", ckpt, "--lora", adapter, "--out", out])
            self.state["artifacts"].append(out.name)
        shutil.rmtree(work / "ckpt", ignore_errors=True)
