#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p weights
cd weights

wget -c https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth
wget -c https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt
wget -c https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt
