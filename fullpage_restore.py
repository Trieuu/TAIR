import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from PIL import Image

import initialize
from terediff.model import ControlLDM, Diffusion
from terediff.sampler import SpacedSampler
from terediff.utils.common import instantiate_from_config


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PATCH_SIZE = 512


def resolve_output_paths(output: str, input_path: Path) -> tuple[Path, Path]:
    output_path = Path(output)
    if output_path.suffix.lower() in IMAGE_SUFFIXES:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path = output_path.with_name(
            f"{output_path.stem}_compare{output_path.suffix}"
        )
        return output_path, compare_path

    output_path.mkdir(parents=True, exist_ok=True)
    restored_path = output_path / f"restored_{input_path.stem}_top_left_512.png"
    compare_path = output_path / f"compare_{input_path.stem}_top_left_512.png"
    return restored_path, compare_path


def resolve_report_dir(output: str) -> Path:
    output_path = Path(output)
    if output_path.suffix.lower() in IMAGE_SUFFIXES:
        return output_path.parent
    return output_path


def count_params(module: nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(p.numel() for p in module.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
    }


def save_param_report(models: dict[str, nn.Module], output: str) -> Path:
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

    report_dir = resolve_report_dir(output)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "param_report.yaml"
    OmegaConf.save(config=OmegaConf.create(report), f=report_path)
    return report_path


def load_top_left_patch(input_path: Path) -> Image.Image:
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width < PATCH_SIZE or height < PATCH_SIZE:
            raise ValueError(
                f"Input image must be at least {PATCH_SIZE}x{PATCH_SIZE}; "
                f"got {width}x{height}: {input_path}"
            )
        return image.crop((0, 0, PATCH_SIZE, PATCH_SIZE))


def restore_patch(args: argparse.Namespace) -> Path:
    input_path = Path(args.input)
    patch = load_top_left_patch(input_path)
    output_path, compare_path = resolve_output_paths(args.output, input_path)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(args.seed, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    gen.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)

    models, _ = initialize.load_model(accelerator, device, args, cfg)

    if args.param_report and accelerator.is_main_process:
        report_path = save_param_report(models, args.output)
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
        val_z, _ = sampler.val_sample(
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
            ts_model=models["testr"],
            val_prompt=val_prompt,
        )

        restored_img = torch.clamp(
            (pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1
        )

    if accelerator.is_main_process:
        restored_img_pil = TF.to_pil_image(restored_img.squeeze().cpu())
        restored_img_pil.save(output_path)
        print(f"Saved restored patch to {output_path}")

        if args.compare:
            compare_img = Image.new("RGB", (PATCH_SIZE * 2, PATCH_SIZE))
            compare_img.paste(patch, (0, 0))
            compare_img.paste(restored_img_pil, (PATCH_SIZE, 0))
            compare_img.save(compare_path)
            print(f"Saved comparison image to {compare_path}")

    accelerator.wait_for_everyone()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore the top-left 512x512 patch of one image with TeReDiff."
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Save original and restored top-left patches side by side.",
    )
    parser.add_argument(
        "--param-report",
        action="store_true",
        help="Save module parameter counts to param_report.yaml in the output folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    restore_patch(parse_args())
