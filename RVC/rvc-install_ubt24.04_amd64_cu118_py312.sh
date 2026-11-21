set -euo pipefail
git clone https://github.com/TianLSama/Retrieval-based-Voice-Conversion-WebUI.git || (cd Retrieval-based-Voice-Conversion-WebUI && git pull --ff-only)
cd ./Retrieval-based-Voice-Conversion-WebUI/
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
python -m pip install "huggingface-hub>=0.34,<1.0" "httpx<0.24" "httpcore<0.17" "h11>=0.14,<0.15" "anyio<4,>=3.6.2" "starlette<0.28" "fastapi<0.100"
python -m pip uninstall -y httpx2 httpcore2 || true
python - <<'PY'
import torch
assert torch.__version__.endswith("+cu118"), f"torch 不是 cu118 构建：{torch.__version__}"
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
PY
python -c "import transformers, gradio, httpx; print('transformers', transformers.__version__, '| gradio', gradio.__version__, '| httpx', httpx.__version__)"
python -m pip check
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" "pretrained/*" "pretrained_v2/*" \
  --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe
hf download lj1995/VoiceConversionWebUI mute.zip --revision main \
  --local-dir .model-downloads
mkdir -p logs
python -m zipfile -e .model-downloads/mute.zip logs
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pymss_weights/*" --local-dir assets
test -f assets/rmvpe/rmvpe.pt
test -f assets/hubert_base/config.json
test -f assets/hubert_base/pytorch_model.bin
test "$(ls assets/pretrained/*.pth | wc -l)" -ge 6
test "$(ls assets/pretrained_v2/*.pth | wc -l)" -ge 6
test "$(ls logs/mute | wc -l)" -ge 1
test "$(ls assets/pymss_weights/*.ckpt | wc -l)" -ge 1
echo "Installation Successful !"
python webui.py --noautoopen
