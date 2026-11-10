#!/usr/bin/env python3
"""Bounded, storage-only TAR prefix experiment. Default is dry-run, no GPU.

Compare the existing buffered tarfile prefix walk with starting at an offset
derived from the v1 JSONL. No production imports, persistent index or cache flush.
This diagnoses read amplification, NOT full-epoch training throughput.
"""
import argparse
import hashlib
import io
import json
import signal
import tarfile
import time
from pathlib import Path


class CountingRaw(io.RawIOBase):
    def __init__(self, path, budget):
        super().__init__()
        self.file = path.open('rb', buffering=0)
        self.budget = budget
        self.bytes = self.reads = self.seeks = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def readinto(self, target):
        if self.budget['remaining'] < len(target):
            raise RuntimeError('Raw-read safety budget reached; no result claimed')
        count = self.file.readinto(target)
        self.budget['remaining'] -= count
        self.bytes += count
        self.reads += 1
        return count

    def seek(self, offset, whence=0):
        self.seeks += 1
        return self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()

    def close(self):
        if not self.closed:
            self.file.close()
        super().close()


def proc_io():
    return {k: int(v) for k, v in
            (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def window(root, shard_index, start, count):
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest.get('format') != 'imagenet-original-webdataset-v1'
            or not manifest.get('complete')):
        raise ValueError('Only complete original USTAR v1 datasets are supported')
    shard = manifest['splits']['train']['shards'][shard_index]
    if not 0 <= start < start + count <= shard['samples']:
        raise ValueError('Requested window is outside shard')
    paths = [(root / shard[k]).resolve() for k in ('path', 'metadata')]
    if any(root not in path.parents for path in paths):
        raise ValueError('Path escapes dataset')
    selected, offset = [], 0
    with paths[1].open() as stream:
        for index, line in enumerate(stream):
            sample = json.loads(line)
            if index >= start:
                selected.append(dict(sample, offset=offset))
            offset += 512 + ((sample['bytes'] + 511) // 512) * 512
            label_size = len(str(sample['label']).encode('ascii'))
            offset += 512 + ((label_size + 511) // 512) * 512
            if index + 1 == start + count:
                break
    if len(selected) != count:
        raise ValueError('Truncated sidecar')
    return paths[0], selected


def measure(path, samples, start, indexed, budget):
    before = proc_io()
    began = time.perf_counter()
    raw = CountingRaw(path, budget)
    useful, digests = 0, []
    with io.BufferedReader(raw, buffer_size=4 * 1024 * 1024) as stream:
        if indexed:
            stream.seek(samples[0]['offset'])
        with tarfile.open(fileobj=stream, mode='r:') as archive:
            members = iter(archive)
            if not indexed:
                for _ in range(start):
                    next(members)
                    next(members)
            for sample in samples:
                image = next(members)
                if image.name != sample['member'] or image.size != sample['bytes']:
                    raise ValueError('Image member/offset mismatch')
                payload = archive.extractfile(image).read()
                label = next(members)
                if label.name != sample['key'] + '.cls':
                    raise ValueError('Label member mismatch')
                if int(archive.extractfile(label).read()) != sample['label']:
                    raise ValueError('Label value mismatch')
                digest = hashlib.sha256(payload).hexdigest()
                if digest != sample['sha256']:
                    raise ValueError('Payload hash mismatch')
                useful += len(payload)
                digests.append(sample['key'] + ':' + digest)
    elapsed = time.perf_counter() - began
    after = proc_io()
    return dict(method='indexed_start' if indexed else 'existing_prefix_walk',
                seconds=elapsed, samples=len(samples), useful_payload_bytes=useful,
                raw_read_bytes=raw.bytes, raw_read_calls=raw.reads, raw_seek_calls=raw.seeks,
                read_amplification=raw.bytes / useful,
                process_io_delta={k: after[k] - before[k] for k in before},
                sample_payload_order_sha256=hashlib.sha256('\n'.join(digests).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=37)
    parser.add_argument('--start', type=int, default=768)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--max-seconds', type=int, default=45)
    parser.add_argument('--max-read-mib', type=int, default=256)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    if not (0 <= args.shard_index and 0 <= args.start and 0 < args.count
            and args.start + args.count <= 1000 and 0 < args.max_seconds <= 60
            and 0 < args.max_read_mib <= 512):
        parser.error('Safety limits: prefix+window<=1000, time<=60s, raw reads<=512MiB')
    if not args.execute:
        print(json.dumps(dict(status='dry_run', root=str(args.root), shard=args.shard_index,
                              start=args.start, count=args.count, max_seconds=args.max_seconds,
                              max_raw_read_mib=args.max_read_mib, cases='baseline/indexed/indexed/baseline'),
                         indent=2))
        return
    if args.result and args.result.exists():
        parser.error('Refuse to overwrite an existing result')

    def expired(signum, frame):
        raise TimeoutError('Time safety limit reached; no complete result claimed')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    root = args.root.resolve()
    path, samples = window(root, args.shard_index, args.start, args.count)
    budget = {'remaining': args.max_read_mib * 1024**2}
    cases = [measure(path, samples, args.start, indexed, budget)
             for indexed in (False, True, True, False)]
    if len({case['sample_payload_order_sha256'] for case in cases}) != 1:
        raise AssertionError('Sample orders differ')
    signal.alarm(0)
    result = dict(status='complete', root=str(root), archive=str(path), start=args.start,
                  count=args.count, cache='uncontrolled; repeated same window likely warm',
                  measurement='storage-only including payload SHA256; no decoding or GPU',
                  limitations='Single-shard prefix diagnosis, not a full-data bandwidth or train speedup',
                  cases=cases, total_raw_read_bytes=args.max_read_mib * 1024**2 - budget['remaining'])
    text = json.dumps(result, indent=2) + '\n'
    if args.result:
        with args.result.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
