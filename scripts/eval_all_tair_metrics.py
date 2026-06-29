#!/usr/bin/env python3
"""
scripts/eval_all_tair_metrics.py

Reproduces the TAIR paper evaluation metrics. The inference pipeline mirrors
val.py exactly (same seed, transforms, model setup, threshold). See comments
throughout for line-by-line correspondence with val.py.

=== What matches the paper ===

  Table 4 — Image Restoration metrics (FULLY REPRODUCIBLE):
    PSNR ↑     pyiqa 'psnr'    (reference-based)
    SSIM ↑     pyiqa 'ssimc'   (val.py uses ssimc, paper calls it SSIM)
    LPIPS ↓    pyiqa 'lpips'   (reference-based)
    DISTS ↓    pyiqa 'dists'   (reference-based)
    FID ↓      pyiqa 'fid'     (restored vs 512×512 preprocessed GT)
    NIQE ↓     pyiqa 'niqe'    (no-reference)
    MANIQA ↑   pyiqa 'maniqa'  (no-reference)
    MUSIQ ↑    pyiqa 'musiq'   (no-reference)
    CLIPIQA ↑  pyiqa 'clipiqa' (no-reference)

=== What does NOT match the paper ===

  Tables 2 & 3 — Text Spotting metrics (APPROXIMATION ONLY):
    The paper evaluates text spotting by running ABCNet v2 [56] and TESTR [101]
    as standard off-the-shelf text spotters on the *restored images*.
    These are image-based models that are separate from the TAIR model.

    This script uses TAIR's fine-tuned TESTR model during diffusion sampling.
    That TESTR takes U-Net features (not raw images), so its predictions are
    an approximation of what a standard image-based TESTR would produce.
    The Det/E2E numbers reported here CANNOT be directly compared to Tables 2/3.

    To exactly reproduce Tables 2/3, save the restored images (done automatically
    to <out_dir>/restored/) and run ABCNet v2 or standard TESTR on them separately.

How it works
------------
TESTR in this codebase takes *diffusion U-Net features* (not raw images), so text
spotting predictions are only available during the diffusion sampling loop.
`sampler.val_sample()` runs TESTR at every denoising timestep and returns
`ts_results`; we use `ts_results[-1]` (final denoising step, most restored features)
for text spotting evaluation.

Usage — smoke test (1 image):
  cd /workspace/TAIR
  python scripts/eval_all_tair_metrics.py \\
    --config        configs/val/local_val_terediff.yaml \\
    --config_testr  testr/configs/TESTR/TESTR_R_50_Polygon.yaml \\
    --ann_json      data/Real-Text/real_benchmark_dataset.json \\
    --out_dir       results/smoke \\
    --max_samples   1

Usage — full Real-Text evaluation:
  cd /workspace/TAIR
  python scripts/eval_all_tair_metrics.py \\
    --config        configs/val/local_val_terediff.yaml \\
    --config_testr  testr/configs/TESTR/TESTR_R_50_Polygon.yaml \\
    --ann_json      data/Real-Text/real_benchmark_dataset.json \\
    --out_dir       results/real_text
"""

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

# TAIR repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "testr"))
import initialize
from adet.evaluation import text_eval_script
from adet.evaluation.text_evaluation import TextEvaluator
from adet.config import get_cfg as get_testr_cfg
from terediff.model import ControlLDM, Diffusion
from terediff.sampler import SpacedSampler
from terediff.utils.common import instantiate_from_config


# ---------------------------------------------------------------------------
# Polygon IoU (mask-based; works for any convex/concave polygon)
# ---------------------------------------------------------------------------

def poly_iou(poly1, poly2, canvas_size=512):
    """Mask-based IoU between two polygons given as lists of (x, y) pairs."""
    m1 = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    m2 = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    pts1 = np.array(poly1, dtype=np.int32).reshape(-1, 1, 2)
    pts2 = np.array(poly2, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m1, [pts1], 1)
    cv2.fillPoly(m2, [pts2], 1)
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Strip, uppercase, keep only alphanumeric + space (CTLABELS-compatible)."""
    return "".join(c for c in text.strip().upper() if c.isalnum() or c == " ")


def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j - 1], dp[j], new_dp[j - 1])
        dp = new_dp
    return dp[n]


def closest_in_lexicon(word: str, lexicon: list) -> str:
    """Return the lexicon entry with minimum edit distance to word."""
    if not lexicon:
        return word
    return min(lexicon, key=lambda w: edit_distance(word, w))


# ---------------------------------------------------------------------------
# Text spotting evaluation
# ---------------------------------------------------------------------------

def evaluate_text_spotting(predictions, annotations, iou_threshold=0.5):
    """
    Compute Det and E2E (None / Full) Precision, Recall, F1.

    predictions : list of dicts  {stem: str, pred_polys: [[x,y]×16], pred_texts: [str]}
    annotations : dict  stem -> {text: [str], poly: [[x,y]×16]}

    Returns dict with Det and E2E metrics.
    """
    # Build full-dataset GT lexicon for "Full" mode
    full_lexicon = sorted({
        normalize_text(t)
        for ann in annotations.values()
        for t in ann["text"]
        if normalize_text(t)
    })

    det_tp = det_fp = det_fn = 0
    e2e_none_tp = e2e_none_fp = e2e_none_fn = 0
    e2e_full_tp = e2e_full_fp = e2e_full_fn = 0

    for pred in predictions:
        stem = pred["stem"]
        if stem not in annotations:
            # No GT for this image — all predictions are FP
            det_fp += len(pred["pred_polys"])
            e2e_none_fp += len(pred["pred_polys"])
            e2e_full_fp += len(pred["pred_polys"])
            continue

        ann = annotations[stem]
        gt_polys = [p for p in ann["poly"]]      # list of 16-point polygons
        gt_texts = [t for t in ann["text"]]       # list of raw GT strings
        n_gt = len(gt_polys)
        n_pred = len(pred["pred_polys"])

        # IoU matching: greedy, one-to-one, highest IoU first
        iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
        for i, pp in enumerate(pred["pred_polys"]):
            for j, gp in enumerate(gt_polys):
                iou_matrix[i, j] = poly_iou(pp, gp)

        matched_gt = [False] * n_gt
        matched_pred = [False] * n_pred
        match_pairs = []  # (pred_idx, gt_idx)

        # Sort all (pred, gt) pairs by IoU descending
        pairs = sorted(
            [(i, j, iou_matrix[i, j]) for i in range(n_pred) for j in range(n_gt)],
            key=lambda x: -x[2],
        )
        for pred_idx, gt_idx, iou in pairs:
            if iou < iou_threshold:
                break
            if matched_pred[pred_idx] or matched_gt[gt_idx]:
                continue
            matched_pred[pred_idx] = True
            matched_gt[gt_idx] = True
            match_pairs.append((pred_idx, gt_idx))

        n_det_match = len(match_pairs)
        det_tp += n_det_match
        det_fp += n_pred - n_det_match
        det_fn += n_gt - n_det_match

        # E2E: for each detection match, also check text recognition
        for pred_idx, gt_idx in match_pairs:
            pred_text_norm = normalize_text(pred["pred_texts"][pred_idx])
            gt_text_norm = normalize_text(gt_texts[gt_idx])

            # None mode: exact case-insensitive match
            none_match = pred_text_norm == gt_text_norm

            # Full mode: snap prediction to nearest lexicon entry, then compare
            if full_lexicon:
                snapped = closest_in_lexicon(pred_text_norm, full_lexicon)
                full_match = snapped == gt_text_norm
            else:
                full_match = none_match

            if none_match:
                e2e_none_tp += 1
            else:
                e2e_none_fp += 1

            if full_match:
                e2e_full_tp += 1
            else:
                e2e_full_fp += 1

        # Unmatched GT instances count as FN for E2E
        unmatched_gt = n_gt - n_det_match
        e2e_none_fn += unmatched_gt
        e2e_full_fn += unmatched_gt
        # Unmatched predictions count as FP for E2E
        unmatched_pred = n_pred - n_det_match
        e2e_none_fp += unmatched_pred
        e2e_full_fp += unmatched_pred

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    return {
        "det": prf(det_tp, det_fp, det_fn),
        "e2e_none": prf(e2e_none_tp, e2e_none_fp, e2e_none_fn),
        "e2e_full": prf(e2e_full_tp, e2e_full_fp, e2e_full_fn),
        "lexicon_size": len(full_lexicon),
        "n_images": len(predictions),
    }


# ---------------------------------------------------------------------------
# Official TESTR text evaluation export
# ---------------------------------------------------------------------------

def _as_sequence(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _looks_like_single_polygon(value) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return False
    if arr.ndim == 1 and arr.size >= 6 and arr.size % 2 == 0:
        return True
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
        return True
    return False


def _annotation_instances(ann: dict) -> list[tuple[object, str]]:
    texts = _as_sequence(ann.get("text"))
    polys_raw = ann.get("poly", [])

    if _looks_like_single_polygon(polys_raw):
        polys = [polys_raw]
    else:
        polys = _as_sequence(polys_raw)

    if not texts:
        texts = ["###"] * len(polys)

    n = min(len(polys), len(texts))
    return [(polys[i], str(texts[i])) for i in range(n)]


def _polygon_points(poly) -> list[tuple[float, float]] | None:
    try:
        arr = np.asarray(poly, dtype=float).reshape(-1, 2)
    except Exception:
        return None

    if len(arr) < 3:
        return None

    points = [(float(x), float(y)) for x, y in arr]
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return None

    # Match TESTR TextEvaluator.sort_detection behavior: invalid polygons are
    # skipped and counter-clockwise rings are reversed before official eval.
    try:
        from shapely.geometry import LinearRing, Polygon

        polygon = Polygon(points)
        if not polygon.is_valid or polygon.area <= 0:
            return None
        if LinearRing(points).is_ccw:
            points = list(reversed(points))
    except Exception:
        pass

    return points


def _format_official_line(poly, text: str) -> str | None:
    points = _polygon_points(poly)
    if not points:
        return None

    coords = []
    for x, y in points:
        coords.extend([str(int(round(x))), str(int(round(y)))])

    # The official parser splits transcription with ',####'.
    clean_text = str(text).replace("\r", " ").replace("\n", " ").strip()
    return ",".join(coords) + ",####" + clean_text


def _write_zip_from_text_files(zip_path: Path, files: dict[str, str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in sorted(files.items()):
            zf.writestr(name, content)


def _cleanup_official_export_dir(eval_dir: Path) -> None:
    import shutil

    for name in ["temp_det_results", "final_temp_det_results"]:
        path = eval_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in ["temp_all_det_cors.txt", "det.zip"]:
        path = eval_dir / name
        if path.exists():
            path.unlink()


def _write_detection_zip_with_testr_export(
    text_results_path: Path,
    eval_dir: Path,
    confidence_threshold: float,
) -> Path:
    """Use TESTR TextEvaluator's own export/sort code to create det.zip."""

    text_results_path = text_results_path.resolve()
    eval_dir = eval_dir.resolve()
    _cleanup_official_export_dir(eval_dir)
    exporter = TextEvaluator.__new__(TextEvaluator)
    cwd = Path.cwd()
    eval_dir.mkdir(parents=True, exist_ok=True)

    try:
        os.chdir(eval_dir)
        exporter.to_eval_format(
            str(text_results_path),
            temp_dir="temp_det_results/",
            cf_th=confidence_threshold,
        )
        det_name = exporter.sort_detection("temp_det_results/")
    finally:
        os.chdir(cwd)

    return eval_dir / det_name


def _parse_official_metric_line(line: str) -> dict[str, float]:
    match = re.search(
        r"precision:\s*([0-9.eE+-]+),\s*recall:\s*([0-9.eE+-]+),\s*hmean:\s*([0-9.eE+-]+)",
        line,
    )
    if not match:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision, recall, hmean = (float(x) for x in match.groups())
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(hmean, 4),
    }


def evaluate_text_spotting_official(
    predictions: list[dict],
    annotations: dict[str, dict],
    out_dir: Path,
    *,
    is_word_spotting: bool,
    confidence_threshold: float,
) -> dict:
    """Run TESTR's bundled official text evaluator on TAIR/TESTR outputs."""

    eval_dir = out_dir / "official_text_eval"
    gt_zip = eval_dir / "gt.zip"
    text_results_path = eval_dir / "text_results.json"
    mapping_path = eval_dir / "sample_mapping.json"

    gt_files: dict[str, str] = {}
    text_results = []
    mapping = []
    n_gt_instances = 0
    n_pred_instances = 0

    for sample_idx, pred in enumerate(predictions, start=1):
        stem = pred["stem"]
        sample_file = f"{sample_idx:07d}.txt"

        ann = annotations.get(stem, {})
        gt_lines = []
        for poly, text in _annotation_instances(ann):
            line = _format_official_line(poly, text)
            if line is not None:
                gt_lines.append(line)
        gt_files[sample_file] = "\n".join(gt_lines)
        n_gt_instances += len(gt_lines)

        pred_polys = pred.get("pred_polys", [])
        pred_texts = pred.get("pred_texts", [])
        pred_scores = pred.get("pred_scores") or [1.0] * len(pred_polys)
        n_instances = min(len(pred_polys), len(pred_texts), len(pred_scores))

        for idx in range(n_instances):
            points = _polygon_points(pred_polys[idx])
            if points is None:
                continue
            text_results.append(
                {
                    "image_id": sample_idx,
                    "category_id": 1,
                    "polys": [[float(x), float(y)] for x, y in points],
                    "rec": str(pred_texts[idx]),
                    "score": float(pred_scores[idx]),
                }
            )
        n_pred_instances += n_instances

        mapping.append(
            {
                "official_sample": sample_file,
                "stem": stem,
                "gt_instances": len(gt_lines),
                "pred_instances": n_instances,
            }
        )

    _write_zip_from_text_files(gt_zip, gt_files)
    eval_dir.mkdir(parents=True, exist_ok=True)
    text_results_path.write_text(json.dumps(text_results), encoding="utf-8")
    det_zip = _write_detection_zip_with_testr_export(
        text_results_path,
        eval_dir,
        confidence_threshold,
    )
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    official = text_eval_script.text_eval_main(
        det_file=str(det_zip),
        gt_file=str(gt_zip),
        is_word_spotting=is_word_spotting,
    )

    return {
        "det": _parse_official_metric_line(official["det_only_method"]),
        "e2e": _parse_official_metric_line(official["e2e_method"]),
        "raw": {
            "det_only_method": official["det_only_method"],
            "e2e_method": official["e2e_method"],
        },
        "files": {
            "gt_zip": str(gt_zip),
            "det_zip": str(det_zip),
            "text_results_json": str(text_results_path),
            "sample_mapping": str(mapping_path),
        },
        "is_word_spotting": is_word_spotting,
        "confidence_threshold": confidence_threshold,
        "n_images": len(predictions),
        "n_gt_instances": n_gt_instances,
        "n_pred_instances": n_pred_instances,
    }


def evaluate_existing_official_text_eval(
    out_dir: Path,
    *,
    is_word_spotting: bool,
    confidence_threshold: float,
) -> dict:
    """Run only the official TESTR evaluator from an existing export folder."""

    eval_dir = out_dir / "official_text_eval"
    gt_zip = eval_dir / "gt.zip"
    text_results_path = eval_dir / "text_results.json"
    mapping_path = eval_dir / "sample_mapping.json"

    missing = [str(path) for path in [gt_zip, text_results_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot run --official_text_eval_only because required file(s) "
            f"are missing: {', '.join(missing)}"
        )

    det_zip = _write_detection_zip_with_testr_export(
        text_results_path,
        eval_dir,
        confidence_threshold,
    )
    official = text_eval_script.text_eval_main(
        det_file=str(det_zip),
        gt_file=str(gt_zip),
        is_word_spotting=is_word_spotting,
    )

    mapping = []
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    text_results = json.loads(text_results_path.read_text(encoding="utf-8"))

    result = {
        "det": _parse_official_metric_line(official["det_only_method"]),
        "e2e": _parse_official_metric_line(official["e2e_method"]),
        "raw": {
            "det_only_method": official["det_only_method"],
            "e2e_method": official["e2e_method"],
        },
        "files": {
            "gt_zip": str(gt_zip),
            "det_zip": str(det_zip),
            "text_results_json": str(text_results_path),
            "sample_mapping": str(mapping_path) if mapping_path.exists() else None,
        },
        "is_word_spotting": is_word_spotting,
        "confidence_threshold": confidence_threshold,
        "n_images": len(mapping) if mapping else None,
        "n_gt_instances": (
            sum(int(item.get("gt_instances", 0)) for item in mapping)
            if mapping
            else None
        ),
        "n_pred_instances": len(text_results),
    }
    metrics_path = eval_dir / "official_metrics.json"
    result["files"]["official_metrics_json"] = str(metrics_path)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------

def check_prerequisites(cfg, config_testr_path, ann_json_path):
    """Abort early with clear messages if any required file is missing."""
    missing = []

    required = {
        "SD weights": cfg.train.sd_path,
        "SwinIR weights": cfg.train.swinir_path,
        "DiffBIR controlnet": cfg.train.resume,
        "TESTR checkpoint": cfg.exp_args.testr_ckpt_dir,
        "TeReDiff stage-3 checkpoint": cfg.exp_args.resume_ckpt_dir,
        "TESTR config": config_testr_path,
        "Annotation JSON": ann_json_path,
        "GT image folder": cfg.dataset.gt_img_path,
        "LQ image folder": cfg.dataset.lq_img_path,
    }

    for label, path in required.items():
        if path and not Path(path).exists():
            missing.append(f"  MISSING  {label}: {path}")

    if missing:
        print("ERROR: required files not found:")
        for m in missing:
            print(m)
        sys.exit(1)

    print("All required files present.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = OmegaConf.load(args.config)
    testr_cfg = get_testr_cfg()
    testr_cfg.merge_from_file(args.config_testr)
    text_eval_confidence = (
        float(args.text_eval_confidence)
        if args.text_eval_confidence is not None
        else float(testr_cfg.MODEL.FCOS.INFERENCE_TH_TEST)
    )

    if args.official_text_eval_only:
        ts_metrics = evaluate_existing_official_text_eval(
            Path(args.out_dir),
            is_word_spotting=args.word_spotting,
            confidence_threshold=text_eval_confidence,
        )
        print("Official TESTR text evaluation completed from existing files.")
        print(f"  Det precision/recall/F1: {ts_metrics['det']}")
        print(f"  E2E precision/recall/F1: {ts_metrics['e2e']}")
        print(f"  Saved: {ts_metrics['files']['official_metrics_json']}")
        return

    # Override config paths if provided via CLI
    if args.gt_img_path:
        cfg.dataset.gt_img_path = args.gt_img_path
    if args.lq_img_path:
        cfg.dataset.lq_img_path = args.lq_img_path
    if args.resume_ckpt:
        cfg.exp_args.resume_ckpt_dir = args.resume_ckpt

    check_prerequisites(cfg, args.config_testr, args.ann_json)

    # Accelerator / device setup — mirrors val.py lines 28-32 exactly.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[ddp_kwargs])
    set_seed(25, device_specific=False)
    device = accelerator.device
    # val.py creates the generator here (line 32), before model loading,
    # so that ordering is preserved even though manual_seed() resets state.
    gen = torch.Generator(device)

    # Output dirs
    out_dir = Path(args.out_dir)
    restored_dir = out_dir / "restored"
    gt_dir_copy = out_dir / "gt"  # symlinks or copies of GT images (for FID)
    restored_dir.mkdir(parents=True, exist_ok=True)
    gt_dir_copy.mkdir(parents=True, exist_ok=True)

    # Load GT annotations
    with open(args.ann_json) as f:
        ann_list = json.load(f)
    ann_by_stem = {Path(a["hq_img"]).stem: a for a in ann_list}

    # Image lists (sorted, matching val.py)
    gt_imgs_path = sorted([
        f"{cfg.dataset.gt_img_path}/{img}"
        for img in os.listdir(cfg.dataset.gt_img_path)
        if img.endswith(".jpg")
    ])
    lq_imgs_path = sorted([
        f"{cfg.dataset.lq_img_path}/{img}"
        for img in os.listdir(cfg.dataset.lq_img_path)
        if img.endswith(".jpg")
    ])
    assert len(gt_imgs_path) == len(lq_imgs_path), \
        f"GT/LQ count mismatch: {len(gt_imgs_path)} vs {len(lq_imgs_path)}"

    print(f"GT image folder: {cfg.dataset.gt_img_path} ({len(gt_imgs_path)} .jpg)")
    print(f"LQ image folder: {cfg.dataset.lq_img_path} ({len(lq_imgs_path)} .jpg)")

    if args.max_samples:
        gt_imgs_path = gt_imgs_path[: args.max_samples]
        lq_imgs_path = lq_imgs_path[: args.max_samples]
        print(f"[smoke test] limiting to {args.max_samples} sample(s)")
    elif len(gt_imgs_path) < 50:
        print(
            "[warn] fewer than 50 images found. If this is not intentional, "
            "the config may still point to demo images instead of converted "
            "dataset folders. Use --gt_img_path and --lq_img_path to override."
        )

    matched_annotations = sum(Path(path).stem in ann_by_stem for path in gt_imgs_path)
    print(f"Images with matching annotations: {matched_annotations}/{len(gt_imgs_path)}")
    if matched_annotations != len(gt_imgs_path):
        print(
            "[warn] some evaluated image stems do not appear in the annotation JSON. "
            "Text metrics for those images will have empty GT. Check --ann_json, "
            "--gt_img_path, and --lq_img_path."
        )

    # Load models via the repo's initialize module (mirrors val.py)
    models, _ = initialize.load_model(accelerator, device, args, cfg)

    # Diffusion / sampler setup (mirrors val.py)
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)

    models = {k: accelerator.prepare(v) for k, v in models.items()}
    pure_cldm: ControlLDM = accelerator.unwrap_model(models["cldm"])

    # IQ metrics (same as val.py; no-reference metrics accept a second arg but ignore it)
    metric_psnr   = pyiqa.create_metric("psnr",    device=device)
    metric_ssim   = pyiqa.create_metric("ssimc",   device=device)
    metric_lpips  = pyiqa.create_metric("lpips",   device=device)
    metric_dists  = pyiqa.create_metric("dists",   device=device)
    metric_niqe   = pyiqa.create_metric("niqe",    device=device)
    metric_musiq  = pyiqa.create_metric("musiq",   device=device)
    metric_maniqa = pyiqa.create_metric("maniqa",  device=device)
    metric_clipiqa = pyiqa.create_metric("clipiqa", device=device)

    # Per-image accumulators
    acc = {k: [] for k in ["psnr", "ssim", "lpips", "dists", "niqe", "musiq", "maniqa", "clipiqa"]}

    # Text spotting predictions collected per image
    ts_predictions = []

    # Preprocessing transforms (mirrors val.py)
    preprocess_gt = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    preprocess_lq = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

    # val.py seeds the generator here (line 89), after metric setup,
    # right before model.eval() — keep that exact ordering.
    gen.manual_seed(25)

    # val.py lines 92-95: put all models in eval mode after accelerator.prepare().
    # initialize.load_model sets cldm to .train() and testr to .train(),
    # so this call is essential — not optional.
    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()

    # val.py line 133 hardcodes models['testr'].test_score_threshold = 0.5
    # inside the loop. We use a CLI arg so it can be overridden for ablations,
    # but default is 0.5 to exactly match val.py.
    testr_sampling_threshold = args.testr_sampling_threshold
    print(f"TESTR sampling threshold (val.py = 0.5): {testr_sampling_threshold}")
    print(f"TESTR official export confidence threshold: {text_eval_confidence}")

    print(f"\nRunning inference on {len(gt_imgs_path)} image(s)...")
    for gt_path, lq_path in tqdm(zip(gt_imgs_path, lq_imgs_path), total=len(gt_imgs_path)):
        gt_id = Path(gt_path).stem
        lq_id = Path(lq_path).stem
        assert gt_id == lq_id, f"Stem mismatch: {gt_id} vs {lq_id}"

        gt_img = Image.open(gt_path)
        lq_img = Image.open(lq_path)

        val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)
        val_lq = preprocess_lq(lq_img).unsqueeze(0).to(device)
        val_bs, _, val_H, val_W = val_gt.shape

        val_prompt = [""]

        with torch.no_grad():
            val_clean = models["swinir"](val_lq)
            val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

            pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)

            models["testr"].test_score_threshold = testr_sampling_threshold
            ts_model = models["testr"]

            val_z, val_ts_results = sampler.val_sample(
                model=models["cldm"],
                device=device,
                steps=50,
                x_size=(val_bs, 4, int(val_H / 8), int(val_W / 8)),
                cond=val_cond,
                uncond=None,
                cfg_scale=1.0,
                x_T=pure_noise,
                progress=accelerator.is_main_process,
                cfg=cfg,
                pure_cldm=pure_cldm,
                ts_model=ts_model,
                val_prompt=val_prompt,
            )

            restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1)
            gt_01 = torch.clamp((val_gt + 1) / 2, 0, 1)

            # Save restored image (needed for FID)
            TF.to_pil_image(restored_img.squeeze().cpu()).save(restored_dir / f"{gt_id}.png")

            # Save GT at 512×512 PNG for FID so both restored and GT images
            # are at identical resolution and format. Copying the original jpg
            # would compare different resolutions if GT is not already 512×512.
            # gt_01 is already the 512×512 preprocessed tensor from val.py's
            # preprocess_gt pipeline (BICUBIC resize → ToTensor → Normalize → clamp).
            gt_dest = gt_dir_copy / f"{gt_id}.png"
            if not gt_dest.exists():
                TF.to_pil_image(gt_01.squeeze().cpu()).save(gt_dest)

            # IQ metrics (identical to val.py)
            acc["psnr"].append(metric_psnr(restored_img, gt_01).mean().item())
            acc["ssim"].append(metric_ssim(restored_img, gt_01).mean().item())
            acc["lpips"].append(metric_lpips(restored_img, gt_01).mean().item())
            acc["dists"].append(metric_dists(restored_img, gt_01).mean().item())
            acc["niqe"].append(metric_niqe(restored_img, gt_01).mean().item())
            acc["musiq"].append(metric_musiq(restored_img, gt_01).mean().item())
            acc["maniqa"].append(metric_maniqa(restored_img, gt_01).mean().item())
            acc["clipiqa"].append(metric_clipiqa(restored_img, gt_01).mean().item())

            # Text spotting: use final denoising step predictions
            if val_ts_results:
                last = val_ts_results[-1]
                ts_predictions.append({
                    "stem": gt_id,
                    "pred_polys": [
                        np.array(p).reshape(-1, 2).tolist() if not isinstance(p, list) else p
                        for p in last["pred_polys"]
                    ],
                    "pred_texts": last["pred_texts"],
                    "pred_scores": last.get("pred_scores", []),
                    "pred_rec_scores": last.get("pred_rec_scores", []),
                })
            else:
                ts_predictions.append({
                    "stem": gt_id,
                    "pred_polys": [],
                    "pred_texts": [],
                    "pred_scores": [],
                    "pred_rec_scores": [],
                })

    # -----------------------------------------------------------------------
    # Aggregate IQ metrics
    # -----------------------------------------------------------------------
    ir_metrics = {k: float(np.mean(v)) for k, v in acc.items()}

    # -----------------------------------------------------------------------
    # FID (skip if too few images — FID is meaningless on <50 images)
    # -----------------------------------------------------------------------
    fid_score = None
    n_images = len(gt_imgs_path)
    if n_images >= 50:
        print(f"\nComputing FID on {n_images} images...")
        try:
            metric_fid = pyiqa.create_metric("fid", device=device)
            fid_score = float(metric_fid(str(restored_dir), str(gt_dir_copy)))
        except Exception as e:
            print(f"[warn] FID computation failed: {e}")
    else:
        print(f"\n[skip] FID requires ≥50 images; have {n_images}. Set --max_samples 0 for full eval.")

    # -----------------------------------------------------------------------
    # Text spotting evaluation
    # -----------------------------------------------------------------------
    print("\nEvaluating text spotting with official TESTR text_eval_script...")
    ts_metrics = evaluate_text_spotting_official(
        ts_predictions,
        ann_by_stem,
        out_dir,
        is_word_spotting=args.word_spotting,
        confidence_threshold=text_eval_confidence,
    )

    legacy_ts_metrics = None
    if args.include_legacy_text_eval:
        legacy_ts_metrics = evaluate_text_spotting(ts_predictions, ann_by_stem)

    # -----------------------------------------------------------------------
    # Collate and save results
    # -----------------------------------------------------------------------
    results = {
        "dataset": args.ann_json,
        "n_images": n_images,
        "image_restoration": {
            "PSNR":    round(ir_metrics["psnr"],    4),
            "SSIM":    round(ir_metrics["ssim"],    4),
            "LPIPS":   round(ir_metrics["lpips"],   4),
            "DISTS":   round(ir_metrics["dists"],   4),
            "FID":     round(fid_score, 4) if fid_score is not None else "N/A (<50 images)",
            "NIQE":    round(ir_metrics["niqe"],    4),
            "MANIQA":  round(ir_metrics["maniqa"],  4),
            "MUSIQ":   round(ir_metrics["musiq"],   4),
            "CLIPIQA": round(ir_metrics["clipiqa"], 4),
        },
        "text_spotting": {
            "official_testr": ts_metrics,
            "note": (
                "Det/E2E are computed by exporting TAIR/TESTR predictions and "
                "converted GT annotations to TESTR's official zip format, then "
                "calling adet.evaluation.text_eval_script.text_eval_main. This "
                "matches the bundled TESTR evaluator path. The E2E value uses "
                "the raw decoded TAIR sampler predictions; lexicon-corrected "
                "Full mode requires a separate TESTR lexicon-matching export."
            ),
        },
    }
    if legacy_ts_metrics is not None:
        results["text_spotting"]["legacy_custom"] = legacy_ts_metrics

    out_json = out_dir / "metrics.json"
    out_txt  = out_dir / "metrics.txt"

    out_json.write_text(json.dumps(results, indent=2))

    lines = [
        "=" * 60,
        f"TAIR Evaluation Results",
        f"Dataset : {args.ann_json}",
        f"Images  : {n_images}",
        "=" * 60,
        "",
        "── Image Restoration ──────────────────────────────────────",
        f"  PSNR    : {results['image_restoration']['PSNR']:>8}  ↑",
        f"  SSIM    : {results['image_restoration']['SSIM']:>8}  ↑",
        f"  LPIPS   : {results['image_restoration']['LPIPS']:>8}  ↓",
        f"  DISTS   : {results['image_restoration']['DISTS']:>8}  ↓",
        f"  FID     : {results['image_restoration']['FID']:>8}  ↓",
        f"  NIQE    : {results['image_restoration']['NIQE']:>8}  ↓",
        f"  MANIQA  : {results['image_restoration']['MANIQA']:>8}  ↑",
        f"  MUSIQ   : {results['image_restoration']['MUSIQ']:>8}  ↑",
        f"  CLIPIQA : {results['image_restoration']['CLIPIQA']:>8}  ↑",
        "",
        "── Text Spotting ──────────────────────────────────────────",
        f"  Backend        : official TESTR text_eval_script.text_eval_main",
        f"  Word spotting  : {ts_metrics['is_word_spotting']}",
        f"  Conf threshold : {ts_metrics['confidence_threshold']}",
        f"  GT instances   : {ts_metrics['n_gt_instances']}",
        f"  Pred instances : {ts_metrics['n_pred_instances']}",
        f"  Det  Precision : {ts_metrics['det']['precision']:>6}",
        f"  Det  Recall    : {ts_metrics['det']['recall']:>6}",
        f"  Det  F1        : {ts_metrics['det']['f1']:>6}",
        f"  E2E  Precision : {ts_metrics['e2e']['precision']:>6}",
        f"  E2E  Recall    : {ts_metrics['e2e']['recall']:>6}",
        f"  E2E  F1        : {ts_metrics['e2e']['f1']:>6}",
        f"  Text results   : {ts_metrics['files']['text_results_json']}",
        f"  Eval GT zip    : {ts_metrics['files']['gt_zip']}",
        f"  Eval det zip   : {ts_metrics['files']['det_zip']}",
        "",
        "=" * 60,
        f"JSON saved to: {out_json}",
    ]
    report = "\n".join(lines)
    out_txt.write_text(report)
    print("\n" + report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TAIR paper metrics evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Val config YAML (e.g. configs/val/local_val_terediff.yaml)",
    )
    parser.add_argument(
        "--config_testr",
        required=True,
        help="TESTR detectron2 config (e.g. testr/configs/TESTR/TESTR_R_50_Polygon.yaml)",
    )
    parser.add_argument(
        "--ann_json",
        required=True,
        help="Annotation JSON from convert_hf_parquet_to_images.py "
             "(e.g. data/Real-Text/real_benchmark_dataset.json)",
    )
    parser.add_argument(
        "--out_dir",
        default="results/eval",
        help="Directory to write results (default: results/eval)",
    )
    parser.add_argument(
        "--gt_img_path",
        default=None,
        help="Override GT image folder from config",
    )
    parser.add_argument(
        "--lq_img_path",
        default=None,
        help="Override LQ image folder from config",
    )
    parser.add_argument(
        "--resume_ckpt",
        default=None,
        help="Override resume_ckpt_dir from config",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Process only the first N images (smoke test). "
             "FID requires ≥50 images and is skipped otherwise.",
    )
    parser.add_argument(
        "--word_spotting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass is_word_spotting to TESTR text_eval_main (default: true).",
    )
    parser.add_argument(
        "--text_eval_confidence",
        type=float,
        default=None,
        help=(
            "Confidence threshold passed to TESTR TextEvaluator.to_eval_format. "
            "Defaults to MODEL.FCOS.INFERENCE_TH_TEST from --config_testr, "
            "matching TESTR's bundled evaluator."
        ),
    )
    parser.add_argument(
        "--testr_sampling_threshold",
        type=float,
        default=0.5,
        help=(
            "TESTR score threshold used during diffusion sampling when OCR "
            "predictions are turned into text prompts (val.py line 133). "
            "Default is 0.5 to exactly match val.py. Changing this alters "
            "the restored image itself. Separate from --text_eval_confidence."
        ),
    )
    parser.add_argument(
        "--include_legacy_text_eval",
        action="store_true",
        help="Also save the previous custom Det/E2E approximation under "
             "text_spotting.legacy_custom for comparison.",
    )
    parser.add_argument(
        "--official_text_eval_only",
        action="store_true",
        help=(
            "Skip model inference/restoration and rerun only TESTR's official "
            "text evaluator from <out_dir>/official_text_eval/text_results.json "
            "and gt.zip."
        ),
    )
    args = parser.parse_args()
    main(args)
