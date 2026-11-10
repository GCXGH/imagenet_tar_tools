#!/usr/bin/env python3
"""Bounded pipeline diagnosis, not a cold-storage or full-training benchmark.

Select <=512 original images, verify TAR/source bytes, retain compressed bytes in
RAM, then isolate decode/augmentation/IPC and optional single-GPU training costs.
No dataset outputs, downloads, model weights, compile, or production imports.
"""
import argparse
import hashlib
import io
import json
import platform
import random
import signal
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, IterableDataset, default_collate, get_worker_info
from torchvision import io as vision_io, models, transforms as T
from torchvision.transforms import functional as F

from loader import encoded_stream, read_manifest, unpack_sample


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def pil_rgb(payload):
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert('RGB')


def make_transform(output):
    # Illustrative ImageNet recipe, not a reproduction of timm's recipe.
    ops = [T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
           T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=9), T.PILToTensor()]
    if output == 'float32':
        ops += [T.ConvertImageDtype(torch.float32), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


class MemoryPipeline(IterableDataset):
    """Same compressed bytes for each run; IPC comparison, not storage benchmark."""
    def __init__(self, records, output, block_size):
        self.records, self.output, self.block_size = records, output, block_size

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        # Per-sample seed makes transform output independent of worker layout.
        transform = make_transform(self.output)
        for start in range(worker * self.block_size, len(self.records), workers * self.block_size):
            batch = []
            for index in range(start, min(start + self.block_size, len(self.records))):
                random.seed(index)
                np.random.seed(index)
                torch.manual_seed(index)
                payload, label, key = self.records[index]
                sample = transform(pil_rgb(payload)), label, key
                if self.block_size == 1:
                    yield sample
                else:
                    batch.append(sample)
            if batch:
                yield default_collate(batch)


def batches(records, output, workers, block_size, batch_size):
    options = dict(batch_size=None, num_workers=workers, pin_memory=False)
    if workers:
        options['prefetch_factor'] = 1
    loader = DataLoader(MemoryPipeline(records, output, block_size), **options)
    iterator = iter(loader)
    try:
        if block_size != 1:
            yield from iterator
        else:
            buffer = []
            for sample in iterator:
                buffer.append(sample)
                if len(buffer) == batch_size:
                    yield default_collate(buffer)
                    buffer = []
            if buffer:
                yield default_collate(buffer)
    finally:
        del iterator


def select_records(root, count):
    root, manifest, shards, digest = read_manifest(root, 'train')
    source = Path(manifest['source']).resolve() / 'train'
    records, selected = [], []
    remaining = count
    # Four existing shards; only short prefixes, never generate a shard subset.
    for shard in shards[:4]:
        n = min((count + 3) // 4, remaining, shard['samples'])
        if n <= 0:
            break
        metadata_path = (root / shard['metadata']).resolve()
        if root not in metadata_path.parents:
            raise ValueError('Metadata escapes dataset')
        with metadata_path.open() as stream:
            rows = [json.loads(next(stream)) for _ in range(n)]
        selected.append((shard, rows))
        remaining -= n
    if remaining:
        raise ValueError('First four shards do not contain enough samples')
    payload_total = sum(row['bytes'] for _, rows in selected for row in rows)
    if payload_total > 128 * 1024**2:
        raise ValueError('Compressed sample RAM budget 128MiB exceeded')
    start = time.perf_counter()
    for shard, rows in selected:
        stream = encoded_stream([str(root / shard['path'])])
        try:
            for row in rows:
                payload, label, key = unpack_sample(next(stream), len(manifest['class_to_idx']))
                if key != row['key'] or label != row['label']:
                    raise ValueError('Key/label mismatch')
                if hashlib.sha256(payload).hexdigest() != row['sha256']:
                    raise ValueError('SHA256 mismatch')
                records.append((payload, label, key))
        finally:
            stream.close()
    tar_seconds = time.perf_counter() - start
    start = time.perf_counter()
    index = 0
    for _, rows in selected:
        for row in rows:
            path = (source / row['source']).resolve()
            if source not in path.parents or path.read_bytes() != records[index][0]:
                raise ValueError('Source payload differs')
            index += 1
    return records, dict(manifest_sha256=digest, samples=count, payload_bytes=payload_total,
                         tar_read_hash_seconds=tar_seconds,
                         source_read_compare_seconds=time.perf_counter()-start,
                         cache='uncontrolled, TAR precedes source; no cold-cache speedup claimed',
                         sample_set_sha256=hashlib.sha256('\n'.join(sorted(r[2] for r in records)).encode()).hexdigest())


def timing(seconds):
    return dict(total_seconds=sum(seconds), p50_seconds=statistics.median(seconds),
                p95_seconds=float(np.percentile(seconds, 95)))


def cpu_probe(records, batch_size):
    decode = []
    for name in ('pil', 'torchvision'):
        start = time.perf_counter()
        pixels, max_diff, mean_diff = 0, 0, []
        for payload, _, _ in records:
            if name == 'pil':
                image = pil_rgb(payload)
                pixels += image.width * image.height
            else:
                image = vision_io.decode_image(torch.frombuffer(bytearray(payload), dtype=torch.uint8),
                                                mode=vision_io.ImageReadMode.RGB)
                pixels += image.shape[1] * image.shape[2]
        seconds = time.perf_counter() - start
        decode.append(dict(decoder=name, seconds=seconds, images_per_second=len(records)/seconds,
                           pixels=pixels, measurement='decode only, compressed input in RAM'))
    # Pixel comparison kept out of decode timing. A diagnostic, not accuracy equivalence.
    for payload, _, _ in records[:32]:
        a = F.pil_to_tensor(pil_rgb(payload)).to(torch.int16)
        b = vision_io.decode_image(torch.frombuffer(bytearray(payload), dtype=torch.uint8),
                                  mode=vision_io.ImageReadMode.RGB).to(torch.int16)
        if a.shape != b.shape:
            raise ValueError('Decoder dimensions differ')
        diff = (a-b).abs()
        max_diff = max(max_diff, int(diff.max()))
        mean_diff.append(float(diff.float().mean()))
    loaders = []
    for output, workers, block_size in [('float32', 0, batch_size), ('float32', 4, 1),
                                         ('float32', 4, batch_size), ('uint8', 4, batch_size)]:
        start = previous = time.perf_counter()
        delays, keys = [], []
        iterator = batches(records, output, workers, block_size, batch_size)
        try:
            for images, labels, ids in iterator:
                delays.append(time.perf_counter()-previous)
                keys.extend(ids)
                previous = time.perf_counter()
        finally:
            iterator.close()
        elapsed = time.perf_counter()-start
        if len(keys) != len(records) or len(set(keys)) != len(records):
            raise ValueError('Sample loss/duplication')
        loaders.append(dict(output=output, workers=workers, block_size=block_size,
                            seconds=elapsed, images_per_second=len(keys)/elapsed,
                            wait=timing(delays), sample_set_sha256=hashlib.sha256(
                                '\n'.join(sorted(keys)).encode()).hexdigest(),
                            includes='startup, PIL, RRC, flip, RandAugment, conversion, IPC, batching; RAM bytes'))
    return dict(decode=decode, pixel_comparison_first_32=dict(max_abs=max_diff,
                mean_abs_per_image=statistics.mean(mean_diff)), loaders=loaders)


def train_probe(records, batch_size, steps, warmup, reverse=False):
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.manual_seed(0)
    model = models.vit_b_16(weights=None).cuda().train()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    mean = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    synthetic = torch.randn(batch_size, 3, 224, 224, device='cuda')
    synthetic_labels = torch.zeros(batch_size, dtype=torch.long, device='cuda')
    cases = []
    configurations = [('synthetic', 'float32', batch_size), ('real_parent', 'float32', 1),
                      ('real_block', 'float32', batch_size), ('real_block_uint8', 'uint8', batch_size)]
    if reverse:
        configurations.reverse()
    for mode, output, block_size in configurations:
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        iterator = None if mode == 'synthetic' else batches(records, output, 4, block_size, batch_size)
        events, waits, losses = [], [], []
        torch.cuda.reset_peak_memory_stats()
        start = None
        try:
            for index in range(warmup + steps):
                if index == warmup:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                begin = time.perf_counter()
                if iterator is None:
                    images, labels = synthetic, synthetic_labels
                else:
                    images, labels, _ = next(iterator)
                wait = time.perf_counter()-begin
                e0, e1, e2 = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                e0.record()
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                if output == 'uint8':
                    images = (images.float().div_(255)-mean)/std
                e1.record()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
                e2.record()
                if index >= warmup:
                    events.append((e0, e1, e2))
                    waits.append(wait)
                    losses.append(loss.detach())
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            if not bool(torch.isfinite(torch.stack(losses)).all()):
                raise ValueError('Nonfinite loss')
            cases.append(dict(mode=mode, seconds_per_update=elapsed/steps,
                              images_per_second=batch_size*steps/elapsed,
                              cpu_batch_wait=timing(waits),
                              gpu_h2d_and_preprocess_ms=statistics.mean(a.elapsed_time(b) for a,b,c in events),
                              gpu_model_step_ms=statistics.mean(b.elapsed_time(c) for a,b,c in events),
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                              finite_losses=True))
        finally:
            if iterator is not None:
                iterator.close()
    return dict(model='torchvision vit_b_16 weights=None, eager BF16, SGD, no DDP/compile',
                batch_size=batch_size, warmup=warmup, measured_steps=steps,
                gpu=torch.cuda.get_device_name(), cases=cases,
                caveat='Synthetic case isolates model but not matching input/labels; real RAM samples, no disk timing. '
                       'CUDA events include stream gaps; CPU wait can overlap GPU, timings not additive.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['cpu', 'train'], default='cpu')
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reverse', action='store_true', help='Reverse GPU case order for repeat')
    args = parser.parse_args()
    if not (0 < args.count <= 512 and 0 < args.batch_size <= 32 and args.count % args.batch_size == 0
            and 0 < args.steps <= 16 and 0 < args.warmup <= 8 and 0 < args.max_seconds <= 240):
        parser.error('Safety limits: count<=512 divisible by batch<=32, steps<=16, warmup<=8, timeout<=240')
    if args.phase == 'train' and (args.steps+args.warmup)*args.batch_size > args.count:
        parser.error('Need enough unique samples for warmup+steps')
    if not args.execute:
        print(json.dumps(dict(status='dry_run', settings=vars(args)), default=str, indent=2))
        return
    if args.result and args.result.exists():
        parser.error('Refuse to overwrite existing result')

    def timeout(signum, frame):
        raise TimeoutError('Probe time budget exceeded')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    torch.set_num_threads(1)
    records, selection = select_records(args.root, args.count)
    result = dict(selection=selection, phase=args.phase, python=platform.python_version(),
                  torch=torch.__version__, recipe='illustrative torchvision RRC224 bicubic + flip + '
                  'RandAugment(2,9) + Tensor + ImageNet normalize; NOT timm recipe equivalence',
                  status='complete', settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    result['measurements'] = (cpu_probe(records, args.batch_size) if args.phase == 'cpu' else
                              train_probe(records, args.batch_size, args.steps, args.warmup, args.reverse))
    signal.alarm(0)
    text = json.dumps(result, indent=2)+'\n'
    if args.result:
        with args.result.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
