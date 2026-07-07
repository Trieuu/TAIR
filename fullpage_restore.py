import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

import initialize
from terediff.model import ControlLDM, Diffusion
from terediff.sampler import SpacedSampler
from terediff.utils.common import instantiate_from_config


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PATCH_SIZE = 512
VIS_FONT_SIZE = 16
VIS_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


def resolve_output_paths(output: str, input_path: Path) -> tuple[Path, Path]:
    output_path = Path(output)
    if output_path.suffix.lower() in IMAGE_SUFFIXES:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir = output_path.parent / input_path.stem
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return output_path, artifact_dir

    artifact_dir = output_path / input_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    restored_path = artifact_dir / f"restored_{input_path.stem}.png"
    return restored_path, artifact_dir


def validate_stride(stride: int) -> None:
    if stride <= 0:
        raise ValueError(f"--stride must be greater than 0; got {stride}")


def iter_patch_coords(width: int, height: int, stride: int):
    x_coords = list(range(0, width, stride))
    for row_idx, y in enumerate(range(0, height, stride)):
        row_x_coords = x_coords if row_idx % 2 == 0 else reversed(x_coords)
        for x in row_x_coords:
            yield x, y


def make_padded_patch(image: Image.Image, x: int, y: int) -> tuple[Image.Image, int, int]:
    width, height = image.size
    valid_width = min(PATCH_SIZE, width - x)
    valid_height = min(PATCH_SIZE, height - y)
    patch = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
    valid_crop = image.crop((x, y, x + valid_width, y + valid_height))
    patch.paste(valid_crop, (0, 0))
    return patch, valid_width, valid_height


def count_params(module: nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(p.numel() for p in module.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
    }


def save_param_report(models: dict[str, nn.Module], report_dir: Path) -> Path:
    module_counts = {
        name: count_params(module)
        for name, module in models.items()
        if isinstance(module, nn.Module)
    }
    report = {
        "total_parameters": sum(
            counts["parameters"] for counts in module_counts.values()
        ),
        "total_trainable_parameters": sum(
            counts["trainable_parameters"] for counts in module_counts.values()
        ),
        "modules": module_counts,
    }

    cldm = models.get("cldm")
    if isinstance(cldm, nn.Module):
        cldm_submodules = {}
        for name in ("unet", "controlnet", "vae", "clip"):
            submodule = getattr(cldm, name, None)
            if isinstance(submodule, nn.Module):
                cldm_submodules[name] = count_params(submodule)
        if cldm_submodules:
            report["cldm_submodules"] = cldm_submodules

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "param_report.yaml"
    OmegaConf.save(config=OmegaConf.create(report), f=report_path)
    return report_path


def load_full_page(input_path: Path) -> Image.Image:
    with Image.open(input_path) as image:
        return image.convert("RGB")


def load_visualization_font(size: int = VIS_FONT_SIZE) -> ImageFont.ImageFont:
    for font_path in VIS_FONT_PATHS:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def draw_testr_visualization(
    restored_patch: Image.Image,
    testr_result: dict | None,
) -> Image.Image:
    visualization = restored_patch.copy()
    if not testr_result:
        return visualization

    draw = ImageDraw.Draw(visualization)
    font = load_visualization_font()
    pred_polys = testr_result.get("pred_polys", [])
    pred_texts = testr_result.get("pred_texts", [])
    pred_scores = testr_result.get("pred_scores", [])

    for polygon, text, score in zip(pred_polys, pred_texts, pred_scores):
        points = [(float(x), float(y)) for x, y in polygon]
        if not points:
            continue

        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        x0 = max(0, min(PATCH_SIZE - 1, int(min(xs))))
        y0 = max(0, min(PATCH_SIZE - 1, int(min(ys))))
        x1 = max(0, min(PATCH_SIZE - 1, int(max(xs))))
        y1 = max(0, min(PATCH_SIZE - 1, int(max(ys))))
        if x1 <= x0 or y1 <= y0:
            continue

        label = f"{score:.3f}: {text}"
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=2)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = x0
        label_y = max(0, y0 - text_height - 4)
        if label_x + text_width + 4 > PATCH_SIZE:
            label_x = max(0, PATCH_SIZE - text_width - 4)

        draw.rectangle(
            (
                label_x,
                label_y,
                min(PATCH_SIZE - 1, label_x + text_width + 4),
                min(PATCH_SIZE - 1, label_y + text_height + 4),
            ),
            fill=(0, 0, 0),
        )
        draw.text((label_x + 2, label_y + 2), label, fill=(0, 255, 0), font=font)

    return visualization


def restore_single_patch(
    patch: Image.Image,
    models: dict[str, nn.Module],
    sampler: SpacedSampler,
    pure_cldm: ControlLDM,
    cfg,
    device: torch.device,
    gen: torch.Generator,
    progress: bool,
) -> tuple[Image.Image, dict | None]:
    val_lq = T.ToTensor()(patch).unsqueeze(0).to(device)
    val_bs, _, val_H, val_W = val_lq.shape
    val_prompt = [""]

    with torch.no_grad():
        val_clean = models["swinir"](val_lq)
        val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)
        pure_noise = torch.randn(
            (1, 4, 64, 64),
            generator=gen,
            device=device,
            dtype=torch.float32,
        )

        models["testr"].test_score_threshold = 0.5
        val_z, ts_results = sampler.val_sample(
            model=models["cldm"],
            device=device,
            steps=50,
            x_size=(val_bs, 4, int(val_H / 8), int(val_W / 8)),
            cond=val_cond,
            uncond=None,
            cfg_scale=1.0,
            x_T=pure_noise,
            progress=progress,
            cfg=cfg,
            pure_cldm=pure_cldm,
            ts_model=models["testr"],
            val_prompt=val_prompt,
        )

        restored_img = torch.clamp(
            (pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1
        )

    final_testr_result = ts_results[-1] if ts_results else None
    return TF.to_pil_image(restored_img.squeeze().cpu()), final_testr_result


def restore_patch(args: argparse.Namespace) -> Path:
    input_path = Path(args.input)
    validate_stride(args.stride)
    original_page = load_full_page(input_path)
    output_path, artifact_dir = resolve_output_paths(args.output, input_path)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(args.seed, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    gen.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)

    models, _ = initialize.load_model(accelerator, device, args, cfg)

    if args.param_report and accelerator.is_main_process:
        report_path = save_param_report(models, artifact_dir)
        print(f"Saved parameter report to {report_path}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )

    models = {k: accelerator.prepare(v) for k, v in models.items()}
    pure_cldm: ControlLDM = accelerator.unwrap_model(models["cldm"])

    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()

    current_result = original_page.copy()
    width, height = original_page.size
    patch_compare_dir = artifact_dir / "patch_compare"
    if args.compare and accelerator.is_main_process:
        patch_compare_dir.mkdir(parents=True, exist_ok=True)

    for patch_idx, (x, y) in enumerate(iter_patch_coords(width, height, args.stride)):
        origin_patch, valid_width, valid_height = make_padded_patch(original_page, x, y)
        input_patch, _, _ = make_padded_patch(current_result, x, y)

        restored_patch, final_testr_result = restore_single_patch(
            input_patch,
            models,
            sampler,
            pure_cldm,
            cfg,
            device,
            gen,
            accelerator.is_main_process,
        )

        valid_restored = restored_patch.crop((0, 0, valid_width, valid_height))
        current_result.paste(valid_restored, (x, y))

        if accelerator.is_main_process:
            print(
                f"Restored patch {patch_idx}: "
                f"x={x}, y={y}, valid={valid_width}x{valid_height}"
            )

            if args.compare:
                visualization_patch = draw_testr_visualization(
                    restored_patch, final_testr_result
                )
                compare_img = Image.new("RGB", (PATCH_SIZE * 4, PATCH_SIZE))
                compare_img.paste(origin_patch, (0, 0))
                compare_img.paste(input_patch, (PATCH_SIZE, 0))
                compare_img.paste(restored_patch, (PATCH_SIZE * 2, 0))
                compare_img.paste(visualization_patch, (PATCH_SIZE * 3, 0))
                patch_compare_path = (
                    patch_compare_dir / f"patch_{patch_idx:04d}_x{x:04d}_y{y:04d}.png"
                )
                compare_img.save(patch_compare_path)

    if accelerator.is_main_process:
        current_result.save(output_path)
        print(f"Saved restored full page to {output_path}")

        if args.compare:
            compare_path = artifact_dir / f"compare_full_{input_path.stem}.png"
            full_compare = Image.new("RGB", (width * 2, height))
            full_compare.paste(original_page, (0, 0))
            full_compare.paste(current_result, (width, 0))
            full_compare.save(compare_path)
            print(f"Saved full-page comparison image to {compare_path}")
            print(f"Saved patch comparison images to {patch_compare_dir}")

    accelerator.wait_for_everyone()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a full image with padded 512x512 TeReDiff patches."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument(
        "--config", type=str, default="configs/val/local_val_terediff.yaml"
    )
    parser.add_argument(
        "--config_testr",
        type=str,
        default="testr/configs/TESTR/TESTR_R_50_Polygon.yaml",
    )
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Save full-page and per-patch comparison images.",
    )
    parser.add_argument(
        "--param-report",
        action="store_true",
        help="Save module parameter counts to param_report.yaml in the output folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    restore_patch(parse_args())
