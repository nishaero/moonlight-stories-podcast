#!/usr/bin/env bash
# Example setup for an AMD Strix Halo (gfx1151) machine; adapt the torch wheel index to your GPU (CUDA users: plain `pip install torch`).
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install -q uv
uv pip install -q --index-strategy unsafe-best-match --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "torch[device-gfx1151]" torchaudio
uv pip install -q -r requirements.txt
sudo apt-get install -y ffmpeg
