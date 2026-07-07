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


def resolve_output_path(output: str, input_path: Path, compare: bool = False) -> Path:
    output_path = Path(output)
    if output_path.suffix.lower() in IMAGE_SUFFIXES:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    prefix = "compare" if compare else "restored"
    return output_path / f"{prefix}_{input_path.stem}_top_left_512.png"


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
    output_path = resolve_output_path(args.output, input_path, args.compare)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(args.seed, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    gen.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)

    models, _ = initialize.load_model(accelerator, device, args, cfg)

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
        if args.compare:
            compare_img = Image.new("RGB", (PATCH_SIZE * 2, PATCH_SIZE))
            compare_img.paste(patch, (0, 0))
            compare_img.paste(restored_img_pil, (PATCH_SIZE, 0))
            compare_img.save(output_path)
            print(f"Saved comparison image to {output_path}")
        else:
            restored_img_pil.save(output_path)
            print(f"Saved restored patch to {output_path}")

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
    return parser.parse_args()


if __name__ == "__main__":
    restore_patch(parse_args())
