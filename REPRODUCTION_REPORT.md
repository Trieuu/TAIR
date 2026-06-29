# TAIR Reproduction Report

**Date:** 2026-06-26  
**Repository:** [TAIR — Text-Aware Image Restoration](https://github.com/TAIR-paper/TAIR)  
**Goal:** Fully reproduce the TAIR paper's experimental setup, including automated model/data download, dataset conversion, and a runnable evaluation script covering all paper metrics.

---

## 1. What Was Done

### 1.1 Repository Audit

Ingested the TAIR repo, paper, and README to map out the full pipeline:

| Component | Description |
|---|---|
| **TeReDiff** | Multi-task diffusion model combining DiffBIR-v2 image restoration with TESTR text spotting |
| **Stage 1** | ControlNet + U-Net attention layers trained for image restoration |
| **Stage 2** | TESTR text spotting module trained on U-Net features |
| **Stage 3** | Both modules fine-tuned jointly |
| **Inference** | `val_sample()` runs TESTR at every denoising timestep; U-Net features feed TESTR, TESTR predictions update the text prompt for the next step |

**Critical architectural detail discovered:** TESTR in TAIR takes *diffusion U-Net extracted features* as input — not raw images. This means text spotting predictions are only available during the diffusion sampling loop, not from a standalone TESTR pass on restored images.

---

### 1.2 Setup Script Fixes (`vast_setup_tair_evals_rerun.sh`)

The original script had several bugs that prevented it from running. All were fixed:

| Bug | Original | Fix |
|---|---|---|
| Wrong SwinIR weight URL | `lxq007/DiffBIR` / `general_full_v1.ckpt` | `lxq007/DiffBIR-v2` / `realesrgan_s4_swinir_100k.pth` |
| Wrong SD base weight URL | `stabilityai/stable-diffusion-2-1-base` / `v2-1_512-ema-pruned.ckpt` | `lxq007/DiffBIR-v2` / `sd2.1-base-zsnr-laionaes5.ckpt` |
| Hardcoded HF token | `export HF_TOKEN="<redacted>"` (plaintext in script) | `HF_TOKEN="${HF_TOKEN:-}"` (injected from environment) |
| TESTR checkpoint not downloadable | Missing from HuggingFace; no download step | Uploaded to Google Drive; added `gdown` download step |
| `hf` / `huggingface-cli` not in PATH | Called directly | Replaced with `uvx --from "huggingface_hub[cli]" hf ...` |
| `uv pip install --system` rejected | Ubuntu externally managed Python blocks `--system` | Removed; all tools called via `uvx` directly |
| Config patching found empty dirs | `first_existing()` returned dirs with no images | Added `any(Path(item).glob("*.jpg"))` guard |
| HF datasets not in expected format | `val.py` expects image folders; HF returns Parquet | Added Parquet → image conversion step |

#### Correct Weight URLs (from the author's own `download_weights.sh`)

```
https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth
https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt
https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt
```

#### TESTR Checkpoint

The TESTR checkpoint (`totaltext_testr_R_50_polygon.pth`) was not distributed via HuggingFace. It was uploaded to Google Drive and the setup script downloads it automatically:

```
Google Drive file ID: 1VuGzGNgnfW9BhnnDvnr5nscAvOV3K1jh
```

---

### 1.3 Weights Downloaded

All 5 required weights are present at `/workspace/TAIR/weights/`:

| File | Size | Source |
|---|---|---|
| `sd2.1-base-zsnr-laionaes5.ckpt` | 4.9 GB | HuggingFace `lxq007/DiffBIR-v2` |
| `terediff_stage3.pt` | 6.4 GB | Google Drive (author-provided) |
| `DiffBIR_v2.1.pt` | 1.4 GB | HuggingFace `lxq007/DiffBIR-v2` |
| `realesrgan_s4_swinir_100k.pth` | 87 MB | HuggingFace `lxq007/DiffBIR-v2` |
| `totaltext_testr_R_50_polygon.pth` | 566 MB | Google Drive (uploaded manually) |

**Total: ~14 GB**

---

### 1.4 Dataset Conversion (`scripts/convert_hf_parquet_to_images.py`)

The two evaluation datasets are distributed on HuggingFace as Parquet files with embedded image bytes. The original `val.py` expects on-disk image folders. A conversion script was written to bridge this gap.

**Script:** `scripts/convert_hf_parquet_to_images.py`  
**Input:** HuggingFace Parquet files downloaded to `data/Real-Text/` and `data/SA-Text-test/`  
**Output:**

```
data/Real-Text/
├── HQ/                          ← 847 ground-truth images (512×512 JPEG)
├── LQ/                          ← 847 low-quality inputs
└── real_benchmark_dataset.json  ← per-image polygon/text annotations

data/SA-Text-test/
├── HQ/                          ← 1000 ground-truth images (512×512 JPEG)
├── LQ_lv1/                      ← 1000 degradation level 1
├── LQ_lv2/                      ← 1000 degradation level 2
├── LQ_lv3/                      ← 1000 degradation level 3
└── sa_text_test_dataset.json    ← per-image polygon/text annotations
```

**Dataset statistics:**

| Dataset | Images | Text Instances |
|---|---|---|
| Real-Text | 847 | 1,857 |
| SA-Text-test | 1,000 | 2,206 |

The annotation JSON format (one entry per image):

```json
{
  "id": "Canon_001_HR_crop_1",
  "hq_img": "HQ/Canon_001_HR_crop_1.jpg",
  "lq_img": "LQ/Canon_001_HR_crop_1.jpg",
  "text": ["SMOKING", "CAMPUS", "ON", "NO"],
  "bbox": [[[112, 409], [261, 440]], ...],
  "poly": [[[112, 411], [133, 411], ..., [114, 439]], ...]
}
```

---

### 1.5 Evaluation Script (`scripts/eval_all_tair_metrics.py`)

A 587-line evaluation script was written that reproduces all metrics reported in the TAIR paper.

#### Paper Metrics Covered

**Image Restoration — Reference-based:**

| Metric | Direction | Implementation |
|---|---|---|
| PSNR | ↑ | `pyiqa.create_metric('psnr')` |
| SSIM | ↑ | `pyiqa.create_metric('ssimc')` |
| LPIPS | ↓ | `pyiqa.create_metric('lpips')` |
| DISTS | ↓ | `pyiqa.create_metric('dists')` |
| FID | ↓ | `pyiqa.create_metric('fid')` on saved image folders |

**Image Restoration — No-reference:**

| Metric | Direction | Implementation |
|---|---|---|
| NIQE | ↓ | `pyiqa.create_metric('niqe')` |
| MANIQA | ↑ | `pyiqa.create_metric('maniqa')` |
| MUSIQ | ↑ | `pyiqa.create_metric('musiq')` |
| CLIPIQA | ↑ | `pyiqa.create_metric('clipiqa')` |

**Text Spotting (TESTR via diffusion U-Net features):**

| Metric | Description |
|---|---|
| Det Precision / Recall / F1 | Polygon IoU ≥ 0.5, one-to-one greedy matching |
| E2E None F1 | Det match + case-insensitive exact text match (no lexicon) |
| E2E Full F1 | Det match + closest word in full-dataset GT lexicon (edit distance) |

#### Design Decisions

1. **No reimplementation of inference.** The script calls `initialize.load_model()` and `sampler.val_sample()` directly from the TAIR repo — the same code path as `val.py`. Only metric aggregation and text spotting evaluation are new.

2. **TESTR predictions from `ts_results[-1]`.** `val_sample()` runs TESTR at every denoising timestep (50 steps total). The final step (`ts_results[-1]`) corresponds to the fully denoised latent, giving the most accurate text predictions. These are compared against GT annotations from the JSON files.

3. **Polygon IoU via mask rasterization.** CV2 `fillPoly` on 512×512 masks gives correct IoU for arbitrary concave polygons without requiring external geometry libraries.

4. **Full lexicon for E2E-Full.** The lexicon is built from all unique normalized GT text instances across the entire dataset — standard word spotting protocol (TotalText/CTW1500 convention).

5. **FID requires ≥50 images.** FID is skipped automatically for small runs (e.g. `--max_samples 1`) and a clear message is printed.

6. **GT coordinates are already 512×512.** Both Real-Text (512×512 crops from RealSR/DrealSR) and SA-Text-test (512×512 crops) are confirmed to have polygon annotations in 512×512 pixel space, matching TESTR's output space.

---

## 2. How to Run

### Prerequisites

```bash
cd /workspace/TAIR
source .venv/bin/activate
```

Verify all weights are present:
```bash
ls -lh weights/
# Should show: DiffBIR_v2.1.pt, DiffBIR_v2.1.pt, realesrgan_s4_swinir_100k.pth,
#              sd2.1-base-zsnr-laionaes5.ckpt, terediff_stage3.pt,
#              totaltext_testr_R_50_polygon.pth
```

### Smoke Test (1 image — ~3 minutes on a single GPU)

```bash
python scripts/eval_all_tair_metrics.py \
  --config        configs/val/local_val_terediff.yaml \
  --config_testr  testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
  --ann_json      data/Real-Text/real_benchmark_dataset.json \
  --out_dir       results/smoke \
  --max_samples   1
```

### Full Real-Text Evaluation (847 images — ~45 min on a single A100)

```bash
python scripts/eval_all_tair_metrics.py \
  --config        configs/val/local_val_terediff.yaml \
  --config_testr  testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
  --ann_json      data/Real-Text/real_benchmark_dataset.json \
  --out_dir       results/real_text
```

### SA-Text-test (1000 images × 3 degradation levels)

```bash
# Level 1
python scripts/eval_all_tair_metrics.py \
  --config        configs/val/local_val_terediff.yaml \
  --config_testr  testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
  --ann_json      data/SA-Text-test/sa_text_test_dataset.json \
  --out_dir       results/sa_text_lv1 \
  --gt_img_path   data/SA-Text-test/HQ \
  --lq_img_path   data/SA-Text-test/LQ_lv1

# Level 2
python scripts/eval_all_tair_metrics.py \
  --config        configs/val/local_val_terediff.yaml \
  --config_testr  testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
  --ann_json      data/SA-Text-test/sa_text_test_dataset.json \
  --out_dir       results/sa_text_lv2 \
  --gt_img_path   data/SA-Text-test/HQ \
  --lq_img_path   data/SA-Text-test/LQ_lv2

# Level 3
python scripts/eval_all_tair_metrics.py \
  --config        configs/val/local_val_terediff.yaml \
  --config_testr  testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
  --ann_json      data/SA-Text-test/sa_text_test_dataset.json \
  --out_dir       results/sa_text_lv3 \
  --gt_img_path   data/SA-Text-test/HQ \
  --lq_img_path   data/SA-Text-test/LQ_lv3
```

---

## 3. Output Format

Each evaluation run produces two files in `--out_dir`:

### `metrics.json`
```json
{
  "dataset": "data/Real-Text/real_benchmark_dataset.json",
  "n_images": 847,
  "image_restoration": {
    "PSNR":    23.45,
    "SSIM":    0.7234,
    "LPIPS":   0.2105,
    "DISTS":   0.1843,
    "FID":     42.17,
    "NIQE":    4.612,
    "MANIQA":  0.4821,
    "MUSIQ":   62.34,
    "CLIPIQA": 0.5913
  },
  "text_spotting": {
    "det":      {"precision": 0.82, "recall": 0.76, "f1": 0.79},
    "e2e_none": {"precision": 0.58, "recall": 0.54, "f1": 0.56},
    "e2e_full": {"precision": 0.63, "recall": 0.59, "f1": 0.61},
    "lexicon_size": 847,
    "n_images": 847
  }
}
```

### `metrics.txt`
Human-readable table with ↑/↓ direction indicators, printed to console and saved.

---

## 4. File Map

```
/workspace/TAIR/
├── vast_setup_tair_evals_rerun.sh         ← automated setup script (fixed)
├── val.py                                 ← original author inference script (unchanged)
├── initialize.py                          ← model loading (unchanged)
├── configs/val/local_val_terediff.yaml    ← val config (generated by setup script)
├── weights/
│   ├── sd2.1-base-zsnr-laionaes5.ckpt    ← 4.9 GB
│   ├── terediff_stage3.pt                ← 6.4 GB  (stage 3 joint checkpoint)
│   ├── DiffBIR_v2.1.pt                   ← 1.4 GB
│   ├── realesrgan_s4_swinir_100k.pth     ← 87 MB
│   └── totaltext_testr_R_50_polygon.pth  ← 566 MB
├── data/
│   ├── Real-Text/
│   │   ├── HQ/                           ← 847 GT images
│   │   ├── LQ/                           ← 847 LQ inputs
│   │   └── real_benchmark_dataset.json   ← annotations
│   └── SA-Text-test/
│       ├── HQ/                           ← 1000 GT images
│       ├── LQ_lv1/ LQ_lv2/ LQ_lv3/     ← 3 degradation levels
│       └── sa_text_test_dataset.json     ← annotations
└── scripts/
    ├── convert_hf_parquet_to_images.py   ← HF Parquet → image folder converter (new)
    └── eval_all_tair_metrics.py          ← full evaluation script (new)
```

---

## 5. Key Technical Findings

### TESTR Takes U-Net Features, Not Images

The most important architectural detail for reproduction: `TransformerDetector.forward(extracted_feats, targets, MODE)` receives intermediate U-Net feature tensors, not images. The image preprocessing path (`preprocess_image`) exists in the class but is bypassed in TAIR's training and inference code. This means:

- TESTR **cannot** be run independently on saved restored images.
- Text spotting evaluation **must** happen inside the diffusion sampling loop.
- The eval script collects `ts_results[-1]` (final denoising step) from `val_sample()`.

### Timestep Ordering

`val_sample()` iterates timesteps in **reverse** (`np.flip(self.timesteps)`), going from high noise (e.g. 999) to low noise (e.g. 0). So:
- `ts_results[0]` = predictions at most-noisy step (worst quality)
- `ts_results[-1]` = predictions at least-noisy step (best quality, used for eval)

### GT Annotation Coordinates

All images in both datasets are 512×512. TESTR predictions are in 512×512 pixel space (hardcoded in `transformer_detector.py`). GT polygon coordinates from the annotation JSONs are also in 512×512 space — no scaling required.

### HuggingFace Parquet Format

The datasets use HuggingFace's structured Parquet format with embedded image bytes and typed array columns (`Array2DExtensionType` for polygons). `polars` cannot read these extension types; `pyarrow.parquet` must be used instead.

---

## 6. Environment

- **Python:** 3.10 (venv at `/workspace/TAIR/.venv`)
- **GPU:** CUDA-capable (tested with A100)
- **Key packages:** `torch`, `torchvision`, `accelerate`, `pyiqa`, `omegaconf`, `detectron2`, `opencv-python`, `pyarrow`, `gdown`
- **Package manager:** `uv` / `uvx` for ephemeral tool invocations
