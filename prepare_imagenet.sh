#!/usr/bin/env bash
set -Eeuo pipefail
root=$(realpath "$(dirname "$0")")
python "$root/prepare.py" \
    --source /mnt/data/xinghaitong/dataset/unpacked_imagenet \
    --output /mnt/data/xinghaitong/dataset/imagenet1k_original_tar_v1 \
    --shard-samples 4096 --seed 0
