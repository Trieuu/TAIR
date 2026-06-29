"""
Convert HuggingFace Parquet dataset downloads to the image-folder structure
expected by the TAIR val.py and training configs.

Output structure
----------------
Real-Text (847 images):
  data/Real-Text/
  ├── HQ/                          <-- gt_img_path in val config
  │   └── *.jpg
  ├── LQ/                          <-- lq_img_path in val config
  │   └── *.jpg
  └── real_benchmark_dataset.json  <-- annotations (bbox, poly, text)

SA-Text-test (1000 images × 3 degradation levels):
  data/SA-Text-test/
  ├── HQ/
  ├── LQ_lv1/
  ├── LQ_lv2/
  ├── LQ_lv3/
  └── sa_text_test_dataset.json

Usage:
  python scripts/convert_hf_parquet_to_images.py [--data-root ./data]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def find_parquet_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.parquet"))


def write_image(img_bytes: bytes | None, out_path: Path) -> bool:
    if not img_bytes:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(img_bytes)
    return True


def count_jpgs(path: Path) -> int:
    return len(list(path.glob("*.jpg"))) if path.exists() else 0


def clear_extracted_outputs(paths: list[Path], ann_path: Path) -> None:
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
    if ann_path.exists():
        ann_path.unlink()


def convert_real_text(parquet_dir: Path, out_dir: Path, overwrite: bool) -> None:
    parquets = find_parquet_files(parquet_dir)
    if not parquets:
        print(f"[skip] no parquet files found in {parquet_dir}")
        return

    hq_dir = out_dir / "HQ"
    lq_dir = out_dir / "LQ"
    ann_path = out_dir / "real_benchmark_dataset.json"

    if (
        not overwrite
        and ann_path.exists()
        and count_jpgs(hq_dir) >= 50
        and count_jpgs(lq_dir) >= 50
    ):
        print(f"[skip] Real-Text already extracted at {out_dir}")
        return

    clear_extracted_outputs([hq_dir, lq_dir], ann_path)

    hq_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow not installed — run: pip install pyarrow")

    annotations = []
    total_written = 0

    for pf_path in parquets:
        print(f"  reading {pf_path.name} ...")
        table = pq.read_table(str(pf_path))
        rows = table.to_pydict()
        n = len(rows["id"])

        for i in range(n):
            img_id = rows["id"][i]
            stem = Path(rows["hq_img"][i]["path"]).stem  # e.g. "Canon_001_HR_crop_1"

            hq_bytes = rows["hq_img"][i]["bytes"]
            lq_bytes = rows["lq_img"][i]["bytes"]

            write_image(hq_bytes, hq_dir / f"{stem}.jpg")
            write_image(lq_bytes, lq_dir / f"{stem}.jpg")
            total_written += 1

            annotations.append({
                "id": img_id,
                "hq_img": f"HQ/{stem}.jpg",
                "lq_img": f"LQ/{stem}.jpg",
                "text": rows["text"][i],
                "bbox": rows["bbox"][i],
                "poly": rows["poly"][i],
            })

    ann_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(f"[done] Real-Text: {total_written} images → {out_dir}")
    print(f"       annotations → {ann_path}")


def convert_sa_text_test(parquet_dir: Path, out_dir: Path, overwrite: bool) -> None:
    parquets = find_parquet_files(parquet_dir)
    if not parquets:
        print(f"[skip] no parquet files found in {parquet_dir}")
        return

    hq_dir = out_dir / "HQ"
    lv_dirs = {
        "lq_img_lv1": out_dir / "LQ_lv1",
        "lq_img_lv2": out_dir / "LQ_lv2",
        "lq_img_lv3": out_dir / "LQ_lv3",
    }
    ann_path = out_dir / "sa_text_test_dataset.json"

    if (
        not overwrite
        and ann_path.exists()
        and count_jpgs(hq_dir) >= 50
        and all(count_jpgs(d) >= 50 for d in lv_dirs.values())
    ):
        print(f"[skip] SA-Text-test already extracted at {out_dir}")
        return

    clear_extracted_outputs([hq_dir, *lv_dirs.values()], ann_path)

    hq_dir.mkdir(parents=True, exist_ok=True)
    for d in lv_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow not installed — run: pip install pyarrow")

    annotations = []
    total_written = 0

    for pf_path in parquets:
        print(f"  reading {pf_path.name} ...")
        table = pq.read_table(str(pf_path))
        rows = table.to_pydict()
        n = len(rows["id"])

        for i in range(n):
            img_id = rows["id"][i]
            stem = Path(rows["hq_img"][i]["path"]).stem

            write_image(rows["hq_img"][i]["bytes"], hq_dir / f"{stem}.jpg")
            total_written += 1

            lq_paths = {}
            for col, lv_dir in lv_dirs.items():
                if col in rows:
                    write_image(rows[col][i]["bytes"], lv_dir / f"{stem}.jpg")
                    lq_paths[col] = f"{lv_dir.name}/{stem}.jpg"

            annotations.append({
                "id": img_id,
                "hq_img": f"HQ/{stem}.jpg",
                **lq_paths,
                "text": rows["text"][i],
                "bbox": rows["bbox"][i],
                "poly": rows["poly"][i],
            })

    ann_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(f"[done] SA-Text-test: {total_written} images (×3 LQ levels) → {out_dir}")
    print(f"       annotations → {ann_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF Parquet datasets to image folders")
    parser.add_argument("--data-root", default="./data", help="Root data directory (default: ./data)")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if output already exists")
    parser.add_argument("--real-text", action="store_true", default=True, help="Convert Real-Text (default: on)")
    parser.add_argument("--sa-text-test", action="store_true", default=True, help="Convert SA-Text-test (default: on)")
    parser.add_argument("--no-real-text", dest="real_text", action="store_false")
    parser.add_argument("--no-sa-text-test", dest="sa_text_test", action="store_false")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.real_text:
        real_parquet_dir = data_root / "Real-Text"
        real_out_dir = data_root / "Real-Text"
        print(f"\n=== Converting Real-Text ===")
        convert_real_text(real_parquet_dir, real_out_dir, args.overwrite)

    if args.sa_text_test:
        sa_parquet_dir = data_root / "SA-Text-test"
        sa_out_dir = data_root / "SA-Text-test"
        print(f"\n=== Converting SA-Text-test ===")
        convert_sa_text_test(sa_parquet_dir, sa_out_dir, args.overwrite)

    print("\n=== All done ===")
    print("Update your val config to point lq_img_path / gt_img_path to the extracted directories.")


if __name__ == "__main__":
    main()
