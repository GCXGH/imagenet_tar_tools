"""Finite WebDataset epochs with explicit coverage/equal-step policies.

No model, training repository or other experimental backend dependencies.
Images stay as original bytes until the caller-supplied PIL transform.
This initial adapter accepts local manifest shards, not arbitrary shell/remote URLs.
"""
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, IterableDataset, default_collate, get_worker_info
import webdataset as wds


IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp'}


def read_manifest(root, split):
    root = Path(root).resolve()
    payload = (root / 'manifest.json').read_bytes()
    manifest = json.loads(payload)
    if not manifest.get('complete'):
        raise ValueError('Require a complete manifest')
    mapping = manifest['class_to_idx']
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError('Class labels must be contiguous from zero')
    shards = manifest['splits'][split]['shards']
    if not shards or len({s['path'] for s in shards}) != len(shards):
        raise ValueError('Empty or duplicate shard list')
    for shard in shards:
        path = (root / shard['path']).resolve()
        if root not in path.parents or shard['samples'] <= 0:
            raise ValueError('Invalid local shard path/count')
    if sum(s['samples'] for s in shards) != manifest['splits'][split]['samples']:
        raise ValueError('Manifest sample counts disagree')
    return root, manifest, shards, hashlib.sha256(payload).hexdigest()


def plan_epoch(shards, world, batch_size, seed=0, epoch=0, shuffle=True, policy='coverage'):
    """Assign whole shards greedily; never split/repeat a shard between ranks.

    coverage: exhaust every shard, partial final rank batch allowed; no equal-step promise.
    equal_steps: trim each rank to min(rank_counts)//batch_size full batches.
    """
    if world < 1 or batch_size < 1 or policy not in ('coverage', 'equal_steps'):
        raise ValueError('Invalid epoch layout/policy')
    ordered = list(shards)
    if shuffle:
        random.Random(seed + epoch).shuffle(ordered)
    groups, counts = [[] for _ in range(world)], [0] * world
    for shard in ordered:
        rank = min(range(world), key=lambda r: (counts[r], r))
        groups[rank].append(shard)
        counts[rank] += shard['samples']
    steps = min(counts) // batch_size
    if policy == 'equal_steps' and steps == 0:
        raise ValueError('Not enough shards/samples for an equal-step epoch')
    limits = [steps * batch_size] * world if policy == 'equal_steps' else counts
    return dict(policy=policy, shards_by_rank=groups, available_by_rank=counts,
                emitted_by_rank=limits, dropped_total=sum(counts) - sum(limits),
                batches_by_rank=[(n + batch_size - 1) // batch_size for n in limits])


def encoded_stream(urls, shuffle_buffer=0, seed=0):
    if not urls:
        return
    # Use WDS's public stages but own local file lifetime explicitly. The tested
    # WebDataset convenience opener emits ResourceWarning on early termination.
    def local_samples():
        for url in urls:
            with open(url, 'rb') as stream:
                expanded = wds.tariterators.tar_file_expander(
                    iter([dict(url=url, stream=stream)]), handler=wds.reraise_exception)
                yield from wds.tariterators.group_by_keys(expanded, handler=wds.reraise_exception)

    source = local_samples()
    pipeline = wds.DataPipeline(lambda: source)
    if shuffle_buffer > 1:
        pipeline.append(wds.shuffle(shuffle_buffer, initial=shuffle_buffer, seed=seed))
    try:
        yield from pipeline
    finally:
        source.close()
        pipeline.close()


def unpack_sample(sample, classes):
    images = [key for key in sample if key in IMAGE_EXTENSIONS]
    if len(images) != 1 or 'cls' not in sample:
        raise ValueError(f"Expected one image and cls: {sample.get('__key__')}")
    label = int(sample['cls'])
    if not 0 <= label < classes:
        raise ValueError('Label out of range')
    return sample[images[0]], label, sample['__key__']


class ShardDataset(IterableDataset):
    def __init__(self, root, shards, classes, transform, seed, epoch, rank, shuffle_buffer):
        self.root, self.shards, self.classes = root, shards, classes
        self.transform, self.seed, self.epoch = transform, seed, epoch
        self.rank, self.shuffle_buffer = rank, shuffle_buffer

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        seed = self.seed + self.epoch * 1000003 + self.rank * 10007 + worker
        if info is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
        shards = self.shards[worker::workers]
        expected = sum(s['samples'] for s in shards)
        count = 0
        urls = [str(self.root / s['path']) for s in shards]
        for sample in encoded_stream(urls, self.shuffle_buffer, seed):
            payload, label, key = unpack_sample(sample, self.classes)
            with Image.open(io.BytesIO(payload)) as image:
                image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            count += 1
            yield image, label, key
        if count != expected:
            raise ValueError(f'Shard count mismatch: expected {expected}, read {count}')


class EpochLoader:
    """Recreate this object for each epoch (no persistent worker epoch state).

    Workers emit individual samples; batching in the parent avoids per-worker
    batch tail loss. This is a correctness-first baseline, not a tuned IPC path.
    Supply a tensor-producing transform for efficient default collation.
    """
    def __init__(self, root, split, transform, batch_size, world=1, rank=0,
                 workers=0, seed=0, epoch=0, shuffle_buffer=512,
                 policy='coverage', prefetch_factor=1, collate_fn=default_collate):
        if not 0 <= rank < world or workers < 0 or shuffle_buffer < 0 or prefetch_factor < 1:
            raise ValueError('Invalid rank/worker/buffer layout')
        root, manifest, shards, digest = read_manifest(root, split)
        plan = plan_epoch(shards, world, batch_size, seed, epoch, split == 'train', policy)
        self.plan, self.manifest_sha256 = plan, digest
        self.limit, self.batch_size = plan['emitted_by_rank'][rank], batch_size
        self.collate_fn = collate_fn
        dataset = ShardDataset(root, plan['shards_by_rank'][rank], len(manifest['class_to_idx']),
                               transform, seed, epoch, rank, shuffle_buffer if split == 'train' else 0)
        options = dict(batch_size=None, num_workers=workers, pin_memory=False)
        if workers:
            options['prefetch_factor'] = prefetch_factor
        self.loader = DataLoader(dataset, **options)

    def __len__(self):
        return (self.limit + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        iterator = iter(self.loader)
        batch = []
        try:
            for _ in range(self.limit):
                try:
                    sample = next(iterator)
                except StopIteration as error:
                    raise ValueError('Stream ended before planned sample count') from error
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []
            if batch:
                yield self.collate_fn(batch)
            if self.plan['policy'] == 'coverage':
                # Exhaust to run count checks; reject unexpected extra samples.
                if next(iterator, None) is not None:
                    raise ValueError('Stream exceeds planned sample count')
        finally:
            # DataLoader owns worker lifecycle; releasing the iterator shuts down
            # non-persistent workers, including when an equal-step epoch is trimmed.
            del iterator
