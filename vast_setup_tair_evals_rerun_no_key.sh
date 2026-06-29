#!/usr/bin/env bash
set -Eeuo pipefail

# Vast/Ubuntu setup for the editable TAIR checkout used by this wiki.
# No secrets are stored here. Pass tokens through environment variables:
#
#   export GITHUB_TOKEN="..."        # optional, for private/fork access
#   export HF_TOKEN="..."            # optional, for gated/private HF assets
#   export WANDB_API_KEY="..."       # optional
#   bash run/vast_setup_tair_evals_rerun_no_key.sh
#
# Useful overrides:
#   PROJECT_DIR=TAIR
#   BRANCH_NAME=trieu/evals-rerun
#   DOWNLOAD_SA_TEXT=true            # full training set; large
#   TESTR_CKPT_URL="https://..."     # optional direct checkpoint URL

echo "[setup] TAIR Vast setup started at $(date)"

# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT_DIR="${PROJECT_DIR:-TAIR}"
REPO_URL="${REPO_URL:-github.com/Trieuu/TAIR.git}"
BRANCH_NAME="${BRANCH_NAME:-trieu/evals-rerun}"

GITHUB_TOKEN="${GITHUB_TOKEN:-}"
HF_TOKEN="${HF_TOKEN:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.2.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.17.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.2.2}"

DOWNLOAD_SA_TEXT="${DOWNLOAD_SA_TEXT:-false}"
DOWNLOAD_SA_TEXT_TEST="${DOWNLOAD_SA_TEXT_TEST:-true}"
DOWNLOAD_REAL_TEXT="${DOWNLOAD_REAL_TEXT:-true}"
DOWNLOAD_TERE_DIFF_STAGE3="${DOWNLOAD_TERE_DIFF_STAGE3:-true}"

# TESTR checkpoint is required for Det/Rec-related validation, but the README
# points to a SharePoint URL rather than a stable scriptable download link.
TESTR_CKPT_URL="${TESTR_CKPT_URL:-}"
TESTR_CKPT_PATH="${TESTR_CKPT_PATH:-weights/totaltext_testr_R_50_polygon.pth}"

TEREDIFF_STAGE3_GDRIVE_URL="${TEREDIFF_STAGE3_GDRIVE_URL:-https://drive.google.com/drive/folders/1Xn0DaL-3ViXpl1pWHPvcmSejTDoIjAQn}"
TEREDIFF_STAGE3_PATH="${TEREDIFF_STAGE3_PATH:-weights/terediff_stage3.pt}"

export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
  echo "[setup] $*"
}

warn() {
  echo "[setup][warn] $*" >&2
}

download_if_missing() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]]; then
    log "exists: $output"
    return 0
  fi
  log "download: $output"
  wget --continue --tries=3 --timeout=30 -O "$output" "$url"
}

remote_content_length() {
  local url="$1"
  curl -L -sI "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:" {size=$2} END {print size}'
}

download_author_weight() {
  local url="$1"
  local output="$2"
  local min_bytes="$3"
  local max_bytes="$4"
  mkdir -p "$(dirname "$output")"

  local local_size=""
  local remote_size=""

  if [[ -e "$output" ]]; then
    local_size="$(stat -c%s "$output" 2>/dev/null || echo 0)"
    remote_size="$(remote_content_length "$url" || true)"

    if [[ -n "$remote_size" && "$local_size" == "$remote_size" ]]; then
      log "exists and matches author download size: $output ($local_size bytes)"
      return 0
    fi

    if [[ "$local_size" -ge "$min_bytes" && "$local_size" -le "$max_bytes" && -z "$remote_size" ]]; then
      log "exists and passes sanity range: $output ($local_size bytes)"
      return 0
    fi

    warn "remove mismatched/partial weight: $output (local=${local_size:-unknown}, remote=${remote_size:-unknown})"
    rm -f "$output"
  fi

  log "download author weight: $output"
  wget --continue --tries=3 --timeout=30 -O "$output" "$url"

  local_size="$(stat -c%s "$output" 2>/dev/null || echo 0)"
  if [[ "$local_size" -lt "$min_bytes" || "$local_size" -gt "$max_bytes" ]]; then
    warn "downloaded $output but size is outside sanity range: $local_size bytes"
  fi
}

hf_download() {
  uvx --from huggingface_hub hf download "$@"
}

extract_archives_in_dir() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  find "$dir" -maxdepth 1 -type f | while read -r archive; do
    case "$archive" in
      *.tar.zst)
        log "extract: $archive"
        tar --use-compress-program=unzstd -xf "$archive" -C "$dir" && rm -f "$archive"
        ;;
      *.tar.gz|*.tgz)
        log "extract: $archive"
        tar -xzf "$archive" -C "$dir" && rm -f "$archive"
        ;;
      *.tar)
        log "extract: $archive"
        tar -xf "$archive" -C "$dir" && rm -f "$archive"
        ;;
      *.zip)
        log "extract: $archive"
        unzip -q -o "$archive" -d "$dir" && rm -f "$archive"
        ;;
      *)
        ;;
    esac
  done
}

find_first_file() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -iname "$pattern" | head -n 1
}

# ---------------------------------------------------------------------------
# System packages and tools
# ---------------------------------------------------------------------------
cd "$WORKSPACE"

log "install system packages"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  ffmpeg \
  fontconfig \
  fonts-hanazono \
  fonts-noto-cjk \
  fonts-unifont \
  git \
  git-lfs \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  nano \
  ninja-build \
  p7zip-full \
  pkg-config \
  python3-dev \
  python3-venv \
  tmux \
  tree \
  unzip \
  wget \
  zip \
  zstd

fc-cache -f || true
git lfs install || true

if [[ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 && ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ]]; then
  ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
fi

if ! command -v uv >/dev/null 2>&1; then
  log "install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [[ -n "$HF_TOKEN" ]]; then
  log "login Hugging Face"
  hf_download --help >/dev/null
  uvx --from huggingface_hub hf auth login --token "$HF_TOKEN" --add-to-git-credential || true
fi

if [[ -n "$WANDB_API_KEY" ]]; then
  log "login Weights & Biases"
  uvx --from wandb wandb login "$WANDB_API_KEY" || true
fi

# ---------------------------------------------------------------------------
# Clone/update TAIR
# ---------------------------------------------------------------------------
if [[ -n "$GITHUB_TOKEN" ]]; then
  CLONE_URL="https://${GITHUB_TOKEN}@${REPO_URL}"
else
  CLONE_URL="https://${REPO_URL}"
fi

if [[ -d "$PROJECT_DIR/.git" ]]; then
  log "update existing repo: $PROJECT_DIR"
  git -C "$PROJECT_DIR" fetch --all --prune
  git -C "$PROJECT_DIR" checkout "$BRANCH_NAME" || git -C "$PROJECT_DIR" checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME"
  git -C "$PROJECT_DIR" pull --ff-only origin "$BRANCH_NAME" || warn "pull failed; keeping local checkout"
else
  log "clone repo branch $BRANCH_NAME"
  git clone --branch "$BRANCH_NAME" "$CLONE_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"
log "repo HEAD: $(git rev-parse --short HEAD) ($(git branch --show-current))"

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
log "create uv environment"
uv venv --python "$PYTHON_VERSION" .venv
source .venv/bin/activate

log "install Python build/runtime dependencies"
# CLIPIQA loads OpenAI CLIP through pyiqa; that CLIP package imports
# pkg_resources, which is no longer guaranteed in newer setuptools releases.
uv pip install --upgrade pip wheel ninja "setuptools<70"
uv pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "$CUDA_WHEEL_INDEX"
uv pip install gdown pyarrow
uv pip install -r requirements.txt
# detectron2/testr setup imports torch during build. uv's default build
# isolation hides the already-installed torch package, so editable installs
# must reuse this venv's build environment.
uv pip install --no-build-isolation -e detectron2
uv pip install --no-build-isolation -e testr

# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------
mkdir -p weights

log "download restoration backbone weights"
download_author_weight \
  "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth" \
  "weights/realesrgan_s4_swinir_100k.pth" \
  1000000 \
  500000000

download_author_weight \
  "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt" \
  "weights/DiffBIR_v2.1.pt" \
  1000000000 \
  2000000000

download_author_weight \
  "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt" \
  "weights/sd2.1-base-zsnr-laionaes5.ckpt" \
  2000000000 \
  8000000000

if [[ -n "$TESTR_CKPT_URL" ]]; then
  log "download TESTR checkpoint from TESTR_CKPT_URL"
  download_if_missing "$TESTR_CKPT_URL" "$TESTR_CKPT_PATH"
else
  warn "TESTR_CKPT_URL is not set. Det/Rec evaluation will need $TESTR_CKPT_PATH manually."
fi

if [[ "$DOWNLOAD_TERE_DIFF_STAGE3" == "true" ]]; then
  if [[ -s "$TEREDIFF_STAGE3_PATH" ]]; then
    log "exists: $TEREDIFF_STAGE3_PATH"
  else
    log "try downloading released TeReDiff checkpoint folder with gdown"
    mkdir -p weights/terediff_stage3_release
    uvx gdown --folder "$TEREDIFF_STAGE3_GDRIVE_URL" -O weights/terediff_stage3_release || warn "gdown folder download failed; download the stage3 checkpoint manually"
    FOUND_STAGE3="$(find_first_file weights/terediff_stage3_release "*.pt" || true)"
    if [[ -n "$FOUND_STAGE3" ]]; then
      cp -f "$FOUND_STAGE3" "$TEREDIFF_STAGE3_PATH"
      log "arranged stage3 checkpoint: $TEREDIFF_STAGE3_PATH"
    else
      warn "No .pt checkpoint found under weights/terediff_stage3_release"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
mkdir -p data

if [[ "$DOWNLOAD_SA_TEXT_TEST" == "true" ]]; then
  log "download SA-Text-test"
  hf_download Min-Jaewon/SA-Text-test \
    --repo-type dataset \
    --local-dir data/SA-Text-test
  extract_archives_in_dir data/SA-Text-test
fi

if [[ "$DOWNLOAD_REAL_TEXT" == "true" ]]; then
  log "download Real-Text"
  hf_download Min-Jaewon/Real-Text \
    --repo-type dataset \
    --local-dir data/Real-Text
  extract_archives_in_dir data/Real-Text
fi

if [[ "$DOWNLOAD_SA_TEXT" == "true" ]]; then
  log "download full SA-Text training data"
  hf_download Min-Jaewon/SA-Text \
    --repo-type dataset \
    --local-dir data/SA-Text
  extract_archives_in_dir data/SA-Text
fi

# ---------------------------------------------------------------------------
# Convert HuggingFace Parquet datasets to the image-folder layout used by TAIR.
# Real-Text becomes data/Real-Text/{HQ,LQ}; SA-Text-test becomes
# data/SA-Text-test/{HQ,LQ_lv1,LQ_lv2,LQ_lv3}.
# ---------------------------------------------------------------------------
log "convert HF Parquet datasets to image folders"
python scripts/convert_hf_parquet_to_images.py --data-root ./data \
  || warn "Parquet conversion failed; val.py/eval scripts need extracted image folders"

# ---------------------------------------------------------------------------
# Local configs and runnable helper scripts
# ---------------------------------------------------------------------------
log "create local config copies"
python - <<'PY'
from pathlib import Path

root = Path(".")

def first_existing(*paths: str) -> str | None:
    for item in paths:
        path = Path(item)
        if path.exists() and any(path.glob("*.jpg")):
            return item
    return None

val_src = root / "configs/val/val_terediff.yaml"
val_dst = root / "configs/val/local_val_terediff.yaml"
val_text = val_src.read_text(encoding="utf-8")

lq_path = first_existing(
    "./data/Real-Text/LQ",
    "./data/Real-Text/lq",
    "./data/SA-Text-test/LQ_lv1",
    "./data/SA-Text-test/LQ",
    "./data/SA-Text-test/lq",
    "./assets/demo_imgs/lq",
)
hq_path = first_existing(
    "./data/Real-Text/HQ",
    "./data/Real-Text/hq",
    "./data/SA-Text-test/HQ",
    "./data/SA-Text-test/hq",
    "./assets/demo_imgs/hq",
)

if lq_path:
    val_text = val_text.replace("lq_img_path: ./assets/demo_imgs/lq", f"lq_img_path: {lq_path}")
if hq_path:
    val_text = val_text.replace("gt_img_path: ./assets/demo_imgs/hq", f"gt_img_path: {hq_path}")

val_text = val_text.replace(
    "resume_ckpt_dir: /PATH/TO/YOUR/STAGE3/CKPT/FILE.pt",
    "resume_ckpt_dir: ./weights/terediff_stage3.pt",
)
val_text = val_text.replace(
    "testr_ckpt_dir:",
    "testr_ckpt_dir: ./weights/totaltext_testr_R_50_polygon.pth",
)
val_dst.write_text(val_text, encoding="utf-8")

train_dir = root / "configs/train"
if train_dir.exists():
    for src in sorted(train_dir.glob("train_stage*_terediff.yaml")):
        dst = train_dir / f"local_{src.name}"
        text = src.read_text(encoding="utf-8")
        text = text.replace("/SET/PATH/TO/sa_text", "./data/SA-Text")
        text = text.replace("/SET/PATH/TO/restoration_dataset.json", "./data/SA-Text/restoration_dataset.json")
        text = text.replace("/PATH/TO/YOUR/STAGE1/CKPT/FILE.pt", "./checkpoints/terediff_stage1.pt")
        text = text.replace("/PATH/TO/YOUR/STAGE2/CKPT/FILE.pt", "./checkpoints/terediff_stage2.pt")
        text = text.replace("testr_ckpt_dir:", "testr_ckpt_dir: ./weights/totaltext_testr_R_50_polygon.pth")
        dst.write_text(text, encoding="utf-8")

print(f"wrote {val_dst}")
PY

mkdir -p run_script/val_script
cat > run_script/val_script/run_val_terediff_local.sh <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
accelerate launch val.py \
  --config configs/val/local_val_terediff.yaml \
  --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml
EOF
chmod +x run_script/val_script/run_val_terediff_local.sh

mkdir -p logs

log "environment summary"
python - <<'PY'
import torch
print("python ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

log "setup complete"
cat <<EOF

Next commands:

  cd $WORKSPACE/$PROJECT_DIR
  source .venv/bin/activate

Smoke-check imports:
  python - <<'PY'
import torch
import detectron2
print(torch.__version__, torch.cuda.is_available())
print("detectron2 import ok")
PY

Run the local TAIR validation/demo config:
  bash run_script/val_script/run_val_terediff_local.sh 2>&1 | tee logs/val_terediff_local.log

Important files:
  local validation config: configs/val/local_val_terediff.yaml
  restoration weights:    weights/
  datasets:               data/

If Det/Rec metrics are needed, make sure this file exists:
  $TESTR_CKPT_PATH

If final TeReDiff inference is needed, make sure this file exists:
  $TEREDIFF_STAGE3_PATH

EOF
