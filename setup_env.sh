#!/bin/bash
# Create the assignment environment in ./.venv (Python 3.11, torch 2.7.1 + CUDA 12.8 wheels). No vLLM or flash-attn.
# Usage:  bash setup_env.sh      then: source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || pip install --user --upgrade uv
UV=$(command -v uv || echo ~/.local/bin/uv)
"$UV" venv --python 3.11 .venv
source .venv/bin/activate
"$UV" pip install -r requirements.txt --torch-backend=cu128
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
