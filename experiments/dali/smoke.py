#!/usr/bin/env python3
"""Bounded DALI CPU/mixed decode, deterministic preprocessing and torch interop.

No model weights, dataset construction, full-shard indexing, or other backend
imports. Original encoded images are loaded once into a <=64MiB RAM buffer.
"""
import argparse
import hashlib
import io
import json
import signal
import statistics
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms as T
from nvidia import dali
from nvidia.dali import fn, types, pipeline_def


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_samples(root, count):
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if not manifest.get('complete'):
        raise ValueError('Incomplete manifest')
    samples = []
    for shard in manifest['splits']['train']['shards'][:4]:
        n = min((count+3)//4, count-len(samples))
        if n <= 0:
            break
        paths = [(root / shard[k]).resolve() for k in ('path', 'metadata')]
        if any(root not in p.parents for p in paths):
            raise ValueError('Manifest path escapes root')
        with paths[1].open() as stream:
            rows = [json.loads(next(stream)) for _ in range(n)]
        with tarfile.open(paths[0], 'r|') as archive:
            members = iter(archive)
            for row in rows:
                image = next(members)
                if image.name != row['member']:
                    raise ValueError('Image member mismatch')
                payload = archive.extractfile(image).read()
                label_member = next(members)
                label = int(archive.extractfile(label_member).read())
                if (label_member.name != row['key']+'.cls' or label != row['label']
                        or hashlib.sha256(payload).hexdigest() != row['sha256']):
                    raise ValueError('Payload/label mismatch')
                samples.append((payload, label, row['key']))
    if len(samples) != count or sum(len(s[0]) for s in samples) > 64*1024**2:
        raise ValueError('Sample count/RAM budget exceeded')
    return samples, hashlib.sha256(manifest_bytes).hexdigest()


class Source:
    def __init__(self, samples, batch_size):
        self.samples, self.batch_size = samples, batch_size

    def __call__(self, info):
        first = info.iteration * self.batch_size
        selected = self.samples[first:first+self.batch_size]
        if len(selected) != self.batch_size:
            raise StopIteration
        return ([np.frombuffer(p, dtype=np.uint8) for p,l,k in selected],
                [np.array([l], dtype=np.int64) for p,l,k in selected])


@pipeline_def
def image_pipeline(source, decoder, preprocess=False):
    encoded, labels = fn.external_source(source=source, num_outputs=2, batch=True,
                                         batch_info=True, dtype=[types.UINT8, types.INT64])
    image = fn.decoders.image(encoded, device=decoder, output_type=types.RGB)
    if preprocess:
        image = fn.resize(image.gpu(), device='gpu', resize_shorter=256,
                          interp_type=types.INTERP_CUBIC, antialias=True)
        image = fn.crop_mirror_normalize(image, device='gpu', dtype=types.FLOAT,
                                        output_layout='CHW', crop=(224,224),
                                        mean=[v*255 for v in MEAN], std=[v*255 for v in STD])
    return image, labels


def build(samples, batch, decoder, preprocess=False):
    return image_pipeline(source=Source(samples,batch), decoder=decoder, preprocess=preprocess,
                          batch_size=batch, num_threads=4, device_id=0, seed=0,
                          prefetch_queue_depth=1)


def torch_batch(pipe):
    images, labels = pipe.run()
    # DALI memory must outlive torch tensors. Caller retains outputs through the step.
    x = torch.from_dlpack(images.as_tensor())
    y = torch.from_numpy(labels.as_array().copy()).reshape(-1).cuda()
    return x, y, (images, labels)


def run(samples, batch):
    decode_results = []
    for decoder in ('cpu', 'mixed'):
        pipe = build(samples,batch,decoder)
        pipe.build()
        diffs, exact, dimensions, first_gpu_ms = [], 0, [], None
        started = time.perf_counter()
        for start in range(0,len(samples),batch):
            images, labels = pipe.run()
            if decoder == 'mixed':
                images = images.as_cpu()
            actual_labels = labels.as_array().reshape(-1).tolist()
            for index in range(batch):
                payload, label, key = samples[start+index]
                if actual_labels[index] != label:
                    raise ValueError('DALI labels differ')
                with Image.open(io.BytesIO(payload)) as image:
                    reference = np.asarray(image.convert('RGB')).astype(np.int16)
                actual = np.asarray(images[index])
                if actual.shape != reference.shape:
                    raise ValueError('Decode dimensions differ')
                difference = np.abs(actual.astype(np.int16)-reference)
                diffs.append((float(difference.mean()),int(difference.max())))
                exact += int(not difference.any())
                dimensions.append(list(actual.shape))
        decode_results.append(dict(decoder=decoder,samples=len(samples),
                                   seconds_including_CPU_copy_and_PIL_comparison=time.perf_counter()-started,
                                   exact_images=exact, mean_abs_per_image=statistics.mean(d[0] for d in diffs),
                                   max_abs=max(d[1] for d in diffs),labels_equal=True,dimensions_equal=True))
        del pipe

    # Compare deterministic preprocessing, NOT random augmentation equivalence.
    validation=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),
                          T.ToTensor(),T.Normalize(MEAN,STD)])
    pipe=build(samples,batch,'mixed',True);pipe.build()
    x,y,owner=torch_batch(pipe)
    reference=torch.stack([validation(Image.open(io.BytesIO(p)).convert('RGB')) for p,l,k in samples[:batch]])
    error=(x.cpu()-reference).abs()
    preprocess=dict(shape=list(x.shape),finite=bool(torch.isfinite(x).all()),
                    normalized_mean_abs=float(error.mean()),normalized_max_abs=float(error.max()),
                    limitation='DALI/PIL bicubic, rounding and center-crop semantics may differ; not bitwise equivalence')
    if x.shape != (batch,3,224,224) or not preprocess['finite']:
        raise ValueError('Invalid preprocessed images')
    del x,y,owner,pipe

    # Minimal complete training loop: startup/cold first step separate from steady steps.
    model=models.vit_b_16(weights=None).cuda().train()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
    pipe=build(samples,batch,'mixed',True);pipe.build()
    times,losses=[],[]
    first=None
    for index in range(min(8,len(samples)//batch)):
        torch.cuda.synchronize();begin=time.perf_counter()
        x,y,owner=torch_batch(pipe)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=torch.nn.functional.cross_entropy(model(x),y)
        loss.backward();optimizer.step();torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        if index==0:first=seconds
        else:times.append(seconds)
        losses.append(float(loss.detach()))
        if not np.isfinite(losses[-1]):
            raise ValueError('Nonfinite training loss')
        del x,y,owner
    return dict(decode=decode_results,preprocess=preprocess,
                train=dict(model='torchvision vit_b_16 weights=None, eager BF16, SGD',
                           input='RAM original bytes -> DALI mixed decode -> GPU deterministic resize/crop/normalize',
                           batch=batch,first_step_seconds=first,subsequent_steps_seconds=times,
                           subsequent_images_per_second=batch/statistics.mean(times),finite_losses=True,
                           caveat='Functional short run only, not same augmentation as preceding WDS benchmark; no speedup ratio'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    if not (32<=args.count<=256 and 1<=args.batch_size<=16 and args.count%args.batch_size==0):
        parser.error('Limits: 32..256 images, batch<=16, exact batches')
    if not args.execute:
        print(json.dumps(dict(status='dry_run',samples=args.count,timeout=240,GPU=1,CPU_threads=4),indent=2));return
    if args.result and args.result.exists():parser.error('Refuse to overwrite result')
    def expired(*args):raise TimeoutError('240-second DALI smoke budget exceeded')
    signal.signal(signal.SIGALRM,expired);signal.alarm(240)
    torch.set_num_threads(1)
    samples,digest=load_samples(args.root.resolve(),args.count)
    result=dict(status='passed',dali=dali.__version__,torch=torch.__version__,manifest_sha256=digest,
                count=len(samples),payload_bytes=sum(len(s[0]) for s in samples),
                measurements=run(samples,args.batch_size))
    signal.alarm(0)
    text=json.dumps(result,indent=2)+'\n'
    if args.result:
        with args.result.open('x') as stream:stream.write(text)
    print(text,end='')


if __name__=='__main__':main()
