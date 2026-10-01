#!/usr/bin/env bash
# One-time setup. Usage:  ./setup.sh          (the duck + Needle Duck Studio: macOS or Linux)
#                         ./setup.sh server   (also the GPU fine-tuning server: Linux + NVIDIA)
set -e
cd "$(dirname "$0")"
if [ ! -d vendor/Open_Duck_Playground ]; then
  git clone -q https://github.com/apirrone/Open_Duck_Playground.git vendor/Open_Duck_Playground
  git -C vendor/Open_Duck_Playground checkout -q b9be205ac64488c23504ca42e5ec790337adeec3
fi
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
if [ "$1" = "server" ] || [ "$1" = "train" ]; then
  pip install -q -r server/requirements.txt
  python -c "import jax; print('JAX devices:', jax.devices())"
fi
python duck_tools.py
mkdir -p models
if [ "$(uname)" = "Darwin" ] && [ -z "$1" ]; then
  ./app/make_mac_app.sh
fi
echo "Done. Start the app with:  .venv/bin/python app/main.py   (or open 'Needle Duck Studio.app' on macOS)"
