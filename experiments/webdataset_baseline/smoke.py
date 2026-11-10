#!/usr/bin/env python3
"""Bounded original-byte/PIL equivalence check; dry-run by default, no GPU."""
import argparse
import hashlib
import io
import json
import signal
import time
from pathlib import Path

from PIL import Image
from loader import encoded_stream, read_manifest, unpack_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    if not 0 < args.count <= 512:
        parser.error('Limit: 1..512 images')
    if not args.execute:
        print(json.dumps(dict(status='dry_run', samples=args.count, timeout_seconds=60,
                              GPU=0, workers=0, writes='small JSON only'), indent=2))
        return
    if args.result and args.result.exists():
        parser.error('Refuse to overwrite an existing result')

    def expired(signum, frame):
        raise TimeoutError('60-second smoke budget exceeded')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    root, manifest, shards, digest = read_manifest(args.root, 'train')
    shard = shards[0]
    if args.count > shard['samples']:
        parser.error('Count exceeds first shard')
    records = []
    with (root / shard['metadata']).open() as stream:
        for _ in range(args.count):
            records.append(json.loads(next(stream)))
    source = Path(manifest['source']).resolve() / 'train'
    started = time.perf_counter()
    samples = encoded_stream([str(root / shard['path'])])
    payload_bytes, keys = 0, []
    try:
        for expected in records:
            payload, label, key = unpack_sample(next(samples), len(manifest['class_to_idx']))
            if label != expected['label'] or key != expected['key']:
                raise ValueError('Sample label/key mismatch')
            if hashlib.sha256(payload).hexdigest() != expected['sha256']:
                raise ValueError('Payload hash mismatch')
            path = (source / expected['source']).resolve()
            if source not in path.parents:
                raise ValueError('Source path escapes root')
            original = path.read_bytes()
            if original != payload:
                raise ValueError('Original bytes differ')
            with Image.open(io.BytesIO(payload)) as image, Image.open(io.BytesIO(original)) as reference:
                image, reference = image.convert('RGB'), reference.convert('RGB')
                if image.size != reference.size or image.tobytes() != reference.tobytes():
                    raise ValueError('PIL RGB differs')
            payload_bytes += len(payload)
            keys.append(key)
    finally:
        samples.close()
    signal.alarm(0)
    result = dict(status='passed', samples=len(keys), unique_samples=len(set(keys)),
                  payload_bytes=payload_bytes, seconds=time.perf_counter()-started,
                  manifest_sha256=digest, original_bytes_equal=True, pil_rgb_equal=True,
                  cache='uncontrolled', GPU=0, workers=0,
                  measurement='correctness only; includes source reread, SHA256 and two PIL decodes',
                  limitation='Not training augmentation equivalence or a throughput benchmark')
    text = json.dumps(result, indent=2) + '\n'
    if args.result:
        with args.result.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
