set -euo pipefail
git clone https://github.com/TianLSama/MSST-WebUI.git || (cd MSST-WebUI && git pull --ff-only)
cd ./MSST-WebUI/
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg libsndfile1 libportaudio2 build-essential curl
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
if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
  echo "SKIP_MODEL_DOWNLOAD=1，跳过模型下载"
else
  echo "开始下载模型（约 3.3GB）..."
  python3 - <<'PY'
import json
import shutil
import subprocess
import sys
from pathlib import Path
TARGETS = [
    ("vocal_models", "model_bs_roformer_ep_368_sdr_12.9628.ckpt"),
    ("vocal_models", "mel_band_roformer_instrumental_becruily.ckpt"),
    ("multi_stem_models", "bs_roformer_4stems_ft.ckpt"),
    ("single_stem_models", "HTDemucs4_FT_drums.th"),
    ("single_stem_models", "HTDemucs4_FT_bass.th"),
    ("single_stem_models", "HTDemucs4_FT_other.th"),
    ("single_stem_models", "HTDemucs4_FT_vocals_official.th"),
    ("single_stem_models", "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"),
]
info_path = Path("data/models_info.json")
if not info_path.is_file():
    info_path = Path("data_backup/models_info.json")
info = json.loads(info_path.read_text(encoding="utf-8"))
if not Path("configs").is_dir():
    shutil.copytree("configs_backup", "configs")
    print("已从 configs_backup 生成 configs 目录", flush=True)
def human(size):
    return f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.1f} MB"
def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sources = [url, url.replace("huggingface.co", "hf-mirror.com")]
    for index, link in enumerate(sources, 1):
        print(f"  下载源 {index}/{len(sources)}: {link}", flush=True)
        cmd = [
            "curl", "-L", "--fail", "--retry", "3", "--retry-delay", "3",
            "--connect-timeout", "20", "-o", str(part),
        ]
        if part.is_file():
            cmd[1:1] = ["-C", "-"]
        cmd.append(link)
        if subprocess.run(cmd).returncode == 0:
            if part.stat().st_size > 0:
                part.replace(dest)
                return True
        print("  该源失败，切换备用源", flush=True)
    return False
failed = []
total = 0
for model_class, name in TARGETS:
    item = info.get(name)
    if not item:
        print(f"模型不在 models_info.json 中: {name}", file=sys.stderr)
        failed.append(name)
        continue
    dest = Path("pretrain") / model_class / name
    expected = int(item.get("model_size") or 0)
    config = Path("configs") / model_class / (name + ".yaml")
    if not config.is_file():
        print(f"警告：缺少配置文件 {config}", file=sys.stderr)
    if dest.is_file() and (expected == 0 or dest.stat().st_size == expected):
        print(f"已存在，跳过: {model_class}/{name} ({human(dest.stat().st_size)})", flush=True)
        total += dest.stat().st_size
        continue
    print(f"下载: {model_class}/{name} ({human(expected)})", flush=True)
    if not download(item["link"], dest):
        failed.append(name)
        continue
    actual = dest.stat().st_size
    if expected and actual != expected:
        print(f"  大小校验失败：期望 {expected}，实际 {actual}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        failed.append(name)
        continue
    print(f"  完成: {human(actual)}", flush=True)
    total += actual
if failed:
    print("以下模型下载失败: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)
print(f"模型下载完成，共 {human(total)}", flush=True)
PY
  echo "================================================================"
  echo "model_bs_roformer_ep_368_sdr_12.9628-----人声"
  echo "mel_band_roformer_instrumental_becruily-----伴奏"
  echo "bs_roformer_4stems_ft.ckpt-----四轨(均衡)"
  echo "HTDemucs4_FT_drums / _bass / _other / _vocals_official----四轨(侧重)：鼓组/贝斯/其他/人声"
  echo "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050-----混响"
  echo "================================================================"
fi
export BROWSER=true
exec python webUI.py -i 0.0.0.0 -p "${MSST_PORT:-7866}"
