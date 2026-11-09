# Original-byte ImageNet TAR preparation

Independent dataset tooling: Python standard library only; no torch, no training
repository imports, no downloads, no image resize, decode or JPEG re-encoding.
Source files remain untouched. Images are copied byte-for-byte into uncompressed
POSIX USTAR archives following the public WebDataset convention: adjacent image
and integer `.cls` members with a shared unique basename.

```bash
python prepare.py --source /path/to/imagenet --output /path/to/imagenet1k_original_tar_v1 \
    --shard-samples 4096 --seed 0
python verify.py /path/to/imagenet1k_original_tar_v1
# Optional: reread original files and compare, in addition to archive verification
python verify.py /path/to/imagenet1k_original_tar_v1 --check-source
python test_prepare.py
```

On machine 61, activate your preferred Python environment and run
`bash prepare_imagenet.sh`. Destination is a sibling of the original dataset:
`/mnt/data/xinghaitong/dataset/imagenet1k_original_tar_v1`.

Classes are sorted exactly as ImageFolder class directories. Supported extensions
match torchvision ImageFolder. Source paths are scanned deterministically, then
shuffled using Python `random.Random(seed)` (val uses seed+1). All samples are
retained, including the final short shard. Training order is NOT prescribed by
this converter: it is the reader's responsibility.

Each shard has a JSONL sidecar recording source path, ID, label and SHA256 of the
original bytes, and a JSON completion record with hashes. `manifest.json` is
published atomically only once both splits complete. Rerunning the same command
resumes completed shards after checking source metadata and archive/sidecar hashes;
settings or converter-source changes require a different output directory.
Only the converter's own `.tmp` files may be overwritten. Do not edit source
data while conversion is running. Do not point at an unrelated populated output.

`verify.py` sequentially checks every image payload hash, label, unique ID and
source path, as well as full shard hashes and counts. It does not decode images;
decoder correctness and DDP sampling belong to downstream reader tests.

## Public references and reproducibility

- WebDataset format and shard writer:
  https://github.com/webdataset/webdataset
  https://github.com/webdataset/webdataset/blob/main/src/webdataset/writer.py
- OpenCLIP epoch-seeded shard shuffle and rank/worker splitting:
  https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/data.py
- NVIDIA DALI mechanical-storage rationale for TAR and buffered sequential reads:
  https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.readers.webdataset.html

These projects were consulted on 2026-11-08. This is an independent implementation
of the public format, not a copy of their source code and not dependent on their
packages. The manifest stores our converter SHA256 and all conversion parameters;
archive metadata (mtime/uid/gid/mode) is fixed. Preserve this entire tools directory
alongside published experiments; it lives outside the training repository and is
not automatically included in that repository's git history.
