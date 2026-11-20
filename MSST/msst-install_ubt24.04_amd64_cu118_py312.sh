set -euo pipefail
git clone https://github.com/TianLSama/MSST-WebUI.git || (cd MSST-WebUI && git pull --ff-only)
cd ./MSST-WebUI/
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg libsndfile1 libportaudio2 build-essential
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python3 - <<'PY'
from pathlib import Path
raw = Path("requirements-uv.txt").read_bytes()
for encoding in ("utf-16", "utf-8-sig", "utf-8"):
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        continue
    if "\x00" not in text:
        break
skip_prefix = ("torch==", "torchaudio==", "torchvision==", "pyside6")
lines = []
for line in text.splitlines():
    pinned = line.strip()
    if not pinned:
        continue
    if pinned.lower().startswith(skip_prefix):
        continue
    if pinned.startswith("samplerate=="):
        pinned = "samplerate==0.2.4"
    lines.append(pinned)
Path("requirements_nocu.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"requirements_nocu.txt: {len(lines)} 个包")
PY
python -m pip install -r requirements_nocu.txt
python -m pip install --force-reinstall --no-deps tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
python - <<'PY'
import torch
assert torch.__version__.endswith("+cu118"), f"torch 不是 cu118 构建：{torch.__version__}"
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
PY
python -c "import torchaudio, torchvision; print('torchaudio', torchaudio.__version__, '| torchvision', torchvision.__version__)"
python -c "import librosa, gradio, transformers, numpy as np, fastapi, pydantic; print('librosa', librosa.__version__, '| gradio', gradio.__version__, '| transformers', transformers.__version__, '| numpy', np.__version__)"
python -m pip check || true
echo "Installation Successful !"
export BROWSER=true
exec python webUI.py -i 0.0.0.0 -p "${MSST_PORT:-7866}"
