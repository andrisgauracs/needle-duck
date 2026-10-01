# needle-train: the fine-tuning server

Fine-tuning Needle 3 wants an NVIDIA GPU. CPU training technically works, but on an M2 Max a single 20-layer epoch takes more than 10 minutes. This small FastAPI service runs on a GPU machine on your network. Needle Duck Studio uploads the training data, streams the log and downloads the finished `.cact` models. Everything else (the sim, inference and scoring) stays on your laptop.

## Setup (Linux + NVIDIA)

```bash
git clone <this repo> ~/needle-duck && cd ~/needle-duck
python3 -m venv .venv && . .venv/bin/activate
pip install -r server/requirements.txt
python -c "import jax; print(jax.devices())"     # must list a CUDA device
```

If `jax.devices()` shows only a CPU, your JAX CUDA wheel is too old for the GPU (common on RTX 50-series cards). Run `pip install -U "jax[cuda12]"` and check again.

Pick a token and start the server:

```bash
mkdir -p ~/needle-train
echo "NEEDLE_API_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(24))')" > ~/needle-train/.env
cat ~/needle-train/.env                          # you'll paste this token into the app
cd server && set -a && . ~/needle-train/.env && set +a
uvicorn train_server:app --host 0.0.0.0 --port 8765
```

Then in the app, go to **Fine-tune → settings** and enter `http://<this machine's LAN IP>:8765` and the token. The status dot turns green when it connects.

To keep it running across reboots, use the systemd template in `needle-train.service` (the setup steps are in its header).

## What a job does

For each requested depth `D` (for example `depths=20,4`):

1. **Slice** (only for D < 20): `slice_base.py` cuts the base checkpoint down to D layers. The result is cached in `~/needle-train/base/needle3_{D}l.safetensors`.
2. **Fine-tune:** `needle finetune train.jsonl --checkpoint <that base> --epochs E --lr LR`
3. **Build:** `needle build <that base> --lora adapter_{D}l.safetensors --out duck_{D}l.cact`

Why each depth gets its own LoRA: `needle finetune` always trains on the full 20 layers, and slicing a 20-layer LoRA afterwards (`needle build --layers N`) produced 8/4/2-layer models whose reasoning looped and that scored below the untuned base.

The first run downloads the Needle 3 base checkpoint (about 240 MB) into `~/needle-train/base/`. JAX's compile cache lives in `~/needle-train/.jax_cache` (several GB). The first run with a new dataset or new settings is a few minutes slower while JAX compiles.

Measured on an RTX 5090 with this repo's dataset (1,717 rows): 20 layers / 10 epochs takes about 12.5 min, and 4 layers / 20 epochs takes about 5 min.

## API

Every endpoint except `/health` needs `Authorization: Bearer <NEEDLE_API_TOKEN>`.

| Method | Path | |
|---|---|---|
| GET | `/health` | `{"ok", "devices", "needle_version", "busy"}` |
| POST | `/jobs` | multipart: `train` (JSONL file), `epochs` (default 10), `lr`, `depths` (default `20`), `name` (default `duck`). Returns `{"job_id"}`. 409 while another job runs |
| GET | `/jobs/latest`, `/jobs/{id}` | status, stage, artifacts, error |
| GET | `/jobs/{id}/logs?offset=N` | `{"text", "offset", "done"}`, polled by the app |
| GET | `/jobs/{id}/artifacts/{file}` | the `.cact` or adapter |
| POST | `/jobs/{id}/cancel` | kills the running step |

Settings: `NEEDLE_API_TOKEN` (if it isn't set, a random token is printed at startup), `NEEDLE_TRAIN_HOME` (default `~/needle-train`), `JAX_COMPILATION_CACHE_DIR`.

The service has no TLS and only a shared token. Keep it on your LAN or behind a VPN, and don't expose it to the internet.
