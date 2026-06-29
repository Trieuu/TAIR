#!/usr/bin/env bash
# ===========================================================================
# run_full_eval.sh — Reproduce ALL TAIR paper evaluations
#
# What this covers
# ----------------
#   Table 4 (image restoration metrics):
#     PSNR, SSIM, LPIPS, DISTS, FID, NIQE, MANIQA, MUSIQ, CLIPIQA
#     → Real-Text  (847 images, real-world HR/LR pairs)
#     → SA-Text-test Level 1, 2, 3  (1000 images × 3 degradation levels)
#
#   Tables 2 & 3 (text spotting via TAIR's TESTR):
#     Det / E2E Precision, Recall, F1
#     NOTE: TAIR's TESTR model takes U-Net features during diffusion sampling,
#     NOT raw images. The paper's Tables 2 & 3 use a SEPARATE standard
#     image-based TESTR run on the finished restored images. The numbers this
#     script produces are therefore an APPROXIMATION of the paper's text
#     spotting columns, but they use the same official TESTR evaluator format.
#     See scripts/eval_testr_on_restored.py for post-hoc image-based eval.
#
# Prerequisites
# -------------
#   cd /workspace/TAIR
#   source .venv/bin/activate
#   # Data already extracted by setup script:
#   #   data/Real-Text/HQ, data/Real-Text/LQ
#   #   data/SA-Text-test/HQ, data/SA-Text-test/LQ_lv1/2/3
#   # If not yet extracted:
#   #   python scripts/convert_hf_parquet_to_images.py --data-root ./data
#
# Usage
# -----
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_eval.sh 2>&1 | tee logs/full_eval.log
#
#   Resume a crashed run (skips already-finished outputs):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_eval.sh --resume
# ===========================================================================
set -Eeuo pipefail
mkdir -p logs results

RESUME=false
for arg in "$@"; do
  [[ "$arg" == "--resume" ]] && RESUME=true
done

CONFIG="configs/val/local_val_terediff.yaml"
CONFIG_TESTR="testr/configs/TESTR/TESTR_R_50_Polygon.yaml"
EVAL_SCRIPT="scripts/eval_all_tair_metrics.py"

run_eval() {
  local label="$1"
  local ann_json="$2"
  local gt_path="$3"
  local lq_path="$4"
  local out_dir="$5"

  local done_marker="${out_dir}/metrics.json"

  if [[ "$RESUME" == "true" && -f "$done_marker" ]]; then
    echo "[skip] ${label} — metrics.json already exists at ${out_dir}"
    return 0
  fi

  echo ""
  echo "================================================================"
  echo " Starting: ${label}"
  echo "   GT  : ${gt_path}"
  echo "   LQ  : ${lq_path}"
  echo "   Out : ${out_dir}"
  echo "================================================================"

  python "$EVAL_SCRIPT" \
    --config "$CONFIG" \
    --config_testr "$CONFIG_TESTR" \
    --ann_json "$ann_json" \
    --gt_img_path "$gt_path" \
    --lq_img_path "$lq_path" \
    --out_dir "$out_dir" \
    --word_spotting

  echo "[done] ${label} → ${out_dir}/metrics.json"
}

# ---------------------------------------------------------------------------
# Real-Text (Table 3 + Table 4 Real-Text row)
# ---------------------------------------------------------------------------
run_eval \
  "Real-Text (Table 3 + Table 4)" \
  "data/Real-Text/real_benchmark_dataset.json" \
  "data/Real-Text/HQ" \
  "data/Real-Text/LQ" \
  "results/paper_real_text"

# ---------------------------------------------------------------------------
# SA-Text-test Level 1 (Table 2 Level1 block + Table 4 SA-Text_test row)
# ---------------------------------------------------------------------------
run_eval \
  "SA-Text-test Level 1 (Table 2 Lv1 + Table 4)" \
  "data/SA-Text-test/sa_text_test_dataset.json" \
  "data/SA-Text-test/HQ" \
  "data/SA-Text-test/LQ_lv1" \
  "results/paper_sa_text_lv1"

# ---------------------------------------------------------------------------
# SA-Text-test Level 2 (Table 2 Level2 block)
# ---------------------------------------------------------------------------
run_eval \
  "SA-Text-test Level 2 (Table 2 Lv2)" \
  "data/SA-Text-test/sa_text_test_dataset.json" \
  "data/SA-Text-test/HQ" \
  "data/SA-Text-test/LQ_lv2" \
  "results/paper_sa_text_lv2"

# ---------------------------------------------------------------------------
# SA-Text-test Level 3 (Table 2 Level3 block)
# ---------------------------------------------------------------------------
run_eval \
  "SA-Text-test Level 3 (Table 2 Lv3)" \
  "data/SA-Text-test/sa_text_test_dataset.json" \
  "data/SA-Text-test/HQ" \
  "data/SA-Text-test/LQ_lv3" \
  "results/paper_sa_text_lv3"

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Summary — compare against paper"
echo "================================================================"
python - <<'PY'
import json
from pathlib import Path

RUNS = [
    ("Real-Text  (Table 3 + Table 4)",  "results/paper_real_text",  "real_text"),
    ("SA-Text Lv1 (Table 2 Lv1)",       "results/paper_sa_text_lv1", "sa_lv1"),
    ("SA-Text Lv2 (Table 2 Lv2)",       "results/paper_sa_text_lv2", "sa_lv2"),
    ("SA-Text Lv3 (Table 2 Lv3)",       "results/paper_sa_text_lv3", "sa_lv3"),
]

# Paper Table 4 reference values (TeReDiff Ours)
PAPER_REF = {
    "Real-Text (Table 4)":
        dict(PSNR=23.37, SSIM=0.7849, LPIPS=0.2848, DISTS=0.2386, FID=68.94,
             NIQE=7.643, MANIQA=0.5637, MUSIQ=62.02, CLIPIQA=0.4545),
    # Table 4 SA-Text_test row is averaged across Lv1/2/3
    "SA-Text-test avg Lv1+2+3 (Table 4)":
        dict(PSNR=19.71, SSIM=0.5717, LPIPS=0.2828, DISTS=0.1702, FID=36.94,
             NIQE=5.452, MANIQA=0.6471, MUSIQ=72.07, CLIPIQA=0.6145),
}

IR_KEYS = ["PSNR", "SSIM", "LPIPS", "DISTS", "FID", "NIQE", "MANIQA", "MUSIQ", "CLIPIQA"]

loaded = {}
for label, out_dir, key in RUNS:
    p = Path(out_dir) / "metrics.json"
    if not p.exists():
        print(f"\n[MISSING] {label}: {p} not found")
        continue
    d = json.loads(p.read_text())
    loaded[key] = d
    ir = d.get("image_restoration", {})
    ts = d.get("text_spotting", {}).get("official_testr", {})
    det = ts.get("det", {})
    e2e = ts.get("e2e", {})
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  Images : {d.get('n_images', '?')}")
    print(f"  Image Restoration:")
    for k in IR_KEYS:
        v = ir.get(k, "N/A")
        print(f"    {k:<8}: {v}")
    print(f"  Text Spotting (TAIR-TESTR approx):")
    print(f"    Det   P/R/F1 : {det.get('precision','?')} / {det.get('recall','?')} / {det.get('f1','?')}")
    print(f"    E2E   P/R/F1 : {e2e.get('precision','?')} / {e2e.get('recall','?')} / {e2e.get('f1','?')}")

# Compute SA-Text average across Lv1/2/3 for Table 4 comparison
sa_keys = ["sa_lv1", "sa_lv2", "sa_lv3"]
sa_loaded = [loaded[k] for k in sa_keys if k in loaded]
if len(sa_loaded) == 3:
    import numpy as np
    print(f"\n{'─'*60}")
    print(f"  SA-Text avg Lv1+2+3 (compare to Table 4 SA-Text_test row):")
    for k in IR_KEYS:
        vals = [d["image_restoration"].get(k) for d in sa_loaded]
        if all(isinstance(v, (int, float)) for v in vals):
            print(f"    {k:<8}: {round(float(np.mean(vals)), 4)}")
        else:
            print(f"    {k:<8}: N/A")

print(f"\n{'─'*60}")
print("  Paper reference (Table 4, TeReDiff Ours):")
for name, ref in PAPER_REF.items():
    print(f"\n  {name}:")
    for k, v in ref.items():
        print(f"    {k:<8}: {v}")
print()
print("NOTE: Tables 2 & 3 text spotting numbers above are APPROXIMATIONS.")
print("  The paper runs a standard image-based TESTR on restored images.")
print("  This script uses TAIR's U-Net-feature TESTR (predictions from")
print("  the final diffusion step). Numbers are NOT directly comparable.")
PY
