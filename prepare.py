#!/usr/bin/env python3
"""Lossless ImageFolder -> WebDataset tar shards; Python standard library only."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tarfile
import time

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
FORMAT = 'imagenet-original-webdataset-v1'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scan(root, mapping):
    samples = []
    for name, label in mapping.items():
        files = []
        for directory, dirs, names in os.walk(root / name, followlinks=True):
            dirs.sort()
            files.extend(Path(directory) / n for n in sorted(names)
                         if Path(n).suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f'Empty class: {root / name}')
        for path in files:
            stat = path.stat()
            samples.append((path.relative_to(root).as_posix(), label, stat.st_size, stat.st_mtime_ns))
    return samples


def add_member(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    info.uid = info.gid = 0
    archive.addfile(info, io.BytesIO(data))


def convert(source, output, shard_samples=4096, seed=0, classes=1000, limit_per_class=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate, non-nested directories')
    if shard_samples < 1 or limit_per_class < 0:
        raise ValueError('Invalid shard size or limit')
    class_names = sorted(p.name for p in (source / 'train').iterdir() if p.is_dir())
    if len(class_names) != classes:
        raise ValueError(f'Expected {classes} classes, got {len(class_names)}')
    if class_names != sorted(p.name for p in (source / 'val').iterdir() if p.is_dir()):
        raise ValueError('Train/val class maps differ')
    mapping = {name: i for i, name in enumerate(class_names)}
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe = {'format': FORMAT, 'source': str(source), 'shard_samples': shard_samples,
                  'seed': seed, 'class_to_idx': mapping, 'limit_per_class': limit_per_class,
                  'converter_sha256': sha256_file(Path(__file__).resolve())}
        recipe_path = output / 'conversion.json'
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
            raise ValueError('Conversion settings/code changed; choose a different output directory')
        if not recipe_path.exists() and any(output.glob('*/**/*.tar')):
            raise ValueError('Unrecognized existing tar output')
        atomic_json(recipe_path, recipe)
        atomic_json(output / 'class_to_idx.json', mapping)
        manifest = dict(recipe, complete=True, splits={})
        for split in ('train', 'val'):
            samples = scan(source / split, mapping)
            if limit_per_class:
                selected = []
                for label in range(classes):
                    group = [s for s in samples if s[1] == label]
                    random.Random(seed + label).shuffle(group)
                    selected.extend(group[:limit_per_class])
                samples = selected
            # Conversion-time mixing prevents class-sorted shards. No image decoding.
            random.Random(seed + (0 if split == 'train' else 1)).shuffle(samples)
            split_root = output / split
            split_root.mkdir(exist_ok=True)
            records = []
            print(f'{split}: {len(samples)} samples', flush=True)
            for shard_index, start in enumerate(range(0, len(samples), shard_samples)):
                chunk = samples[start:start + shard_samples]
                name = f'{split}-{shard_index:06d}.tar'
                target = split_root / name
                metadata_path = target.with_suffix('.jsonl')
                record_path = target.with_suffix('.json')
                chunk_hash = hashlib.sha256(json.dumps(chunk).encode()).hexdigest()
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    if (record['source_list_sha256'] != chunk_hash or not target.is_file()
                            or target.stat().st_size != record['bytes']
                            or not metadata_path.is_file()
                            or sha256_file(metadata_path) != record['metadata_sha256']
                            or sha256_file(target) != record['sha256']):
                        raise ValueError(f'Cannot resume changed/corrupt shard: {target}')
                    records.append(record)
                    print(f'reused {split}/{name}', flush=True)
                    continue
                started = time.monotonic()
                temporary = target.with_suffix('.tar.tmp')
                meta_tmp = metadata_path.with_suffix('.jsonl.tmp')
                with temporary.open('wb') as raw, meta_tmp.open('w') as metadata:
                    with tarfile.open(fileobj=raw, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                        for offset, (relative, label, size, mtime) in enumerate(chunk):
                            path = source / split / relative
                            data = path.read_bytes()
                            stat = path.stat()
                            if len(data) != size or stat.st_size != size or stat.st_mtime_ns != mtime:
                                raise ValueError(f'Source changed during conversion: {path}')
                            key = f'{start + offset:09d}'
                            member = key + path.suffix.lower()
                            add_member(archive, member, data)
                            add_member(archive, key + '.cls', str(label).encode('ascii'))
                            metadata.write(json.dumps({'key': key, 'member': member,
                                'source': relative, 'label': label, 'bytes': size,
                                'sha256': hashlib.sha256(data).hexdigest()}) + '\n')
                    raw.flush()
                    os.fsync(raw.fileno())
                    metadata.flush()
                    os.fsync(metadata.fileno())
                os.replace(temporary, target)
                os.replace(meta_tmp, metadata_path)
                record = {'path': f'{split}/{name}', 'samples': len(chunk),
                          'bytes': target.stat().st_size, 'sha256': sha256_file(target),
                          'metadata': f'{split}/{metadata_path.name}',
                          'metadata_sha256': sha256_file(metadata_path),
                          'source_list_sha256': chunk_hash}
                atomic_json(record_path, record)
                records.append(record)
                print(f'wrote {split}/{name}: {len(chunk)} samples, '
                      f'{record["bytes"] / 2**20:.1f} MiB, {time.monotonic()-started:.1f}s', flush=True)
            manifest['splits'][split] = {'samples': len(samples), 'shards': records}
        # Publish only after BOTH splits are complete. Readers require this manifest.
        atomic_json(output / 'manifest.json', manifest)
        print(f'COMPLETE: {output / "manifest.json"}', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard-samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--classes', type=int, default=1000)
    parser.add_argument('--limit-per-class', type=int, default=0,
                        help='Diagnostic subset only; 0 preserves every sample')
    args = parser.parse_args()
    convert(args.source, args.output, args.shard_samples, args.seed, args.classes, args.limit_per_class)
