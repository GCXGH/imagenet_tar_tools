#!/usr/bin/env python3
"""Full-epoch DDP pipeline benchmark; dry-run unless --execute is supplied.

DALI implementation is standalone. An explicit read-only training adapter uses
the requested model/step and original loader for a faithful production reference;
no implementation is imported from other experiments or changed in that repo.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms as T
from nvidia.dali import fn, types, pipeline_def
from nvidia.dali.auto_aug.rand_augment import rand_augment
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]


def manifest(root):
    data=(root/'manifest.json').read_bytes()
    m=json.loads(data)
    if m.get('format')!='imagenet-original-webdataset-v1' or not m.get('complete') or m.get('limit_per_class'):
        raise ValueError('Full original-byte USTAR v1 required')
    return m,hashlib.sha256(data).hexdigest()


def make_indexes(root, destination):
    """Derive v1.2 DALI indexes from known USTAR layout; never scan image bytes.

    All entries verified against payloads during prior full dataset verification;
    short reader cross-check precedes any benchmark. Atomic new files only.
    """
    m,digest=manifest(root)
    destination.mkdir(parents=True,exist_ok=True)
    signature=destination/'manifest.sha256'
    if signature.exists():
        if signature.read_text().strip()!=digest:
            raise ValueError('Index manifest changed')
        if all((destination/(Path(s['path']).stem+'.idx')).exists() for s in m['splits']['train']['shards']):
            return
        raise ValueError('Incomplete index directory; use a new destination')
    if any(destination.iterdir()):
        raise ValueError('Refuse unrecognized nonempty index destination')
    for shard in m['splits']['train']['shards']:
        metadata=(root/shard['metadata']).resolve()
        if root not in metadata.parents:
            raise ValueError('Metadata path escapes root')
        offset,count=0,0
        path=destination/(Path(shard['path']).stem+'.idx')
        with path.open('x') as output, metadata.open() as source:
            output.write(f"v1.2 {shard['samples']}\n")
            for line in source:
                sample=json.loads(line)
                extension=sample['member'].rsplit('.',1)[1]
                image_offset=offset+512
                offset=image_offset+((sample['bytes']+511)//512)*512
                label_offset=offset+512
                label_size=len(str(sample['label']).encode('ascii'))
                offset=label_offset+((label_size+511)//512)*512
                output.write(f"{extension} {image_offset} {sample['bytes']} {sample['member']} "
                             f"cls {label_offset} {label_size} {sample['key']}.cls\n")
                count+=1
        if count!=shard['samples'] or offset>shard['bytes']:
            raise ValueError('Invalid sidecar layout/count')
    signature.write_text(digest+'\n')


def label_from_ascii(label):
    length=label.shape()[0]
    digits=fn.cast(fn.pad(label,axes=[0],shape=[3],fill_value=48),dtype=types.INT64)-48
    return ((length==1)*digits[0]+(length==2)*(digits[0]*10+digits[1])+
            (length==3)*(digits[0]*100+digits[1]*10+digits[2]))


@pipeline_def(enable_conditionals=True)
def dali_pipeline(paths,indexes,rank,world,decoder,augment,shuffle,reader_io):
    image,label=fn.readers.webdataset(paths=paths,index_paths=indexes,
        ext=['jpg;jpeg;png;ppm;bmp;pgm;tif;tiff;webp','cls'],
        missing_component_behavior='error',name='Reader',num_shards=world,shard_id=rank,
        random_shuffle=shuffle,initial_fill=512,pad_last_batch=False,
        dont_use_mmap=reader_io=='plain',read_ahead=False)
    label=label_from_ascii(label)
    if augment:
        image=fn.decoders.image_random_crop(image,device=decoder,output_type=types.RGB,
              random_area=[0.08,1.0],random_aspect_ratio=[0.75,4/3],num_attempts=10,
              device_memory_padding=16*1024**2,host_memory_padding=4*1024**2)
        image=fn.resize(image.gpu(),device='gpu',resize_x=224,resize_y=224,
                        interp_type=types.INTERP_CUBIC,antialias=True)
        mirror=fn.random.coin_flip(probability=0.5)
        image=fn.flip(image,device='gpu',horizontal=mirror)
        image=rand_augment(image,n=2,m=9,shape=(224,224),fill_value=128,
                           interp_type=types.INTERP_LINEAR,monotonic_mag=True)
        image=fn.crop_mirror_normalize(image,device='gpu',dtype=types.FLOAT,output_layout='CHW',
                                       mean=[v*255 for v in MEAN],std=[v*255 for v in STD])
    return image,label.gpu() if augment else label


def build_dali(args,rank,world,augment=True,shuffle=True):
    m,_=manifest(args.root)
    shards=list(m['splits']['train']['shards'])
    if shuffle:random.Random(args.seed+args.epoch).shuffle(shards)
    paths=[str(args.root/s['path']) for s in shards]
    indexes=[str(args.index_dir/(Path(s['path']).stem+'.idx')) for s in shards]
    pipe=dali_pipeline(paths=paths,indexes=indexes,rank=rank,world=world,decoder=args.decoder,
        augment=augment,shuffle=shuffle,reader_io=args.reader_io,
        batch_size=args.batch_size,num_threads=args.workers,device_id=int(os.environ.get('LOCAL_RANK','0')),
        seed=args.seed+args.epoch*100003+rank,prefetch_queue_depth=1)
    pipe.build()
    return pipe


def reader_check(args):
    """Check native reader bytes/labels for one batch without decode/model."""
    pipe=build_dali(args,0,1,augment=False,shuffle=False)
    actual,labels=pipe.run()
    m,_=manifest(args.root)
    shard=m['splits']['train']['shards'][0]
    with (args.root/shard['metadata']).open() as source:
        for i in range(args.batch_size):
            row=json.loads(next(source))
            image=np.asarray(actual[i])
            if hashlib.sha256(image.tobytes()).hexdigest()!=row['sha256']:
                raise ValueError('Native reader/index payload mismatch')
            if int(np.asarray(labels[i]))!=row['label']:
                raise ValueError('Native reader ASCII label mismatch')
    print(json.dumps(dict(status='reader_check_passed',samples=args.batch_size,reader_meta=pipe.reader_meta()),indent=2))


def percentile(values,p):
    return float(np.percentile(values,p)) if values else 0


def main(args):
    if not args.execute:
        print(json.dumps(dict(status='dry_run',settings=vars(args)),default=str,indent=2));return
    if args.prepare_indexes:
        make_indexes(args.root,args.index_dir);return
    if args.reader_check:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
        reader_check(args);return
    rank,world,local=[int(os.environ.get(k,v)) for k,v in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')]]
    if args.result.exists():raise ValueError('Refuse existing result')
    torch.cuda.set_device(local);torch.set_num_threads(1)
    device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',device_id=device)
    random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
    # Explicit benchmark-only adapter. Production code is read-only and hashed below.
    import sys
    sys.path.insert(0,str(args.training_root))
    train=importlib.import_module('train');models=importlib.import_module('models')
    from timm.loss import SoftTargetCrossEntropy
    from timm.data.random_erasing import RandomErasing
    config=train.read_config(args.config);m,t=config['model'],config['train']
    data_manifest,data_digest=manifest(args.root)
    total=data_manifest['splits']['train']['samples'];full_steps=total//(world*args.batch_size)
    steps=args.steps or full_steps
    if not 0<steps<=full_steps:raise ValueError('Invalid steps')
    if args.batch_size*world!=1024 and not args.steps:
        raise ValueError('Full epoch requires global batch1024')
    start_setup=time.perf_counter()
    raw=models.ClassificationViT(m).to(device).train()
    optimizer=torch.optim.AdamW(models.parameter_groups(raw,t['weight_decay']),lr=t['lr'],
                                 betas=tuple(t['betas']),eps=t['eps'])
    model=DDP(raw,device_ids=[local],broadcast_buffers=False) if world>1 else raw
    forward=train.training_forward(model,args.compile_mode)
    mixup=train.DeviceAwareMixup(mixup_alpha=t['mixup'],cutmix_alpha=t['cutmix'],
        prob=t['mixup_probability'],switch_prob=t['switch_probability'],mode='batch',
        label_smoothing=t['label_smoothing'],num_classes=m['classes'])
    criterion=SoftTargetCrossEntropy()
    erase=RandomErasing(probability=t['re_probability'],mode='pixel',max_count=1,device=str(device))
    if args.backend=='dali':
        pipe=build_dali(args,rank,world)
        iterator=DALIGenericIterator([pipe],['images','labels'],reader_name='Reader',
                auto_reset=False,last_batch_policy=LastBatchPolicy.DROP,prepare_first_batch=False)
        if len(iterator)!=full_steps:
            raise ValueError(f'DALI planned {len(iterator)} steps, expected {full_steps}; metadata={pipe.reader_meta()}')
    else:
        transform=train.make_training_transform(m,t)
        if args.backend=='pil_matched':
            transform=T.Compose([T.RandomResizedCrop(224,interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),T.RandAugment(2,9,interpolation=T.InterpolationMode.BILINEAR,fill=128),
                T.ToTensor(),T.Normalize(MEAN,STD)])
        data=train.OriginalTarDataset(args.root,'train',transform,args.batch_size,world=world,rank=rank,
                seed=args.seed,shuffle_buffer=512)
        data.set_epoch(args.epoch)
        loader=DataLoader(data,batch_size=args.batch_size,num_workers=args.workers,prefetch_factor=1,pin_memory=False)
        if len(loader)!=full_steps:raise ValueError('Original loader step count mismatch')
        iterator=iter(loader)
    torch.cuda.synchronize()
    if world>1:dist.barrier()
    setup_seconds=time.perf_counter()-start_setup
    waits,latencies,window_seconds,window_samples=[],[],[],[]
    seen_labels=torch.zeros(m['classes'],device=device,dtype=torch.int64)
    torch.cuda.reset_peak_memory_stats()
    started=last_window=time.perf_counter();first_seconds=None;first_end=None
    for step in range(steps):
        begin=time.perf_counter()
        if args.backend=='dali':
            batch=next(iterator)[0];images,labels=batch['images'],batch['labels'].reshape(-1).long()
        else:
            images,labels=next(iterator)
            images,labels=images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
        waits.append(time.perf_counter()-begin)
        if args.backend!='original':images=erase(images)
        seen_labels+=torch.bincount(labels,minlength=m['classes'])
        images,targets=mixup(images,labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=criterion(forward(images).float(),targets)
        train.assert_finite(loss,'benchmark loss');loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),t['clip_grad']);train.assert_finite(norm,'benchmark grad')
        optimizer.step()
        # Loss/grad checks synchronize like production; window end additionally synchronizes all ranks.
        latencies.append(time.perf_counter()-begin)
        if step==0:
            torch.cuda.synchronize();first_seconds=time.perf_counter()-started;first_end=time.perf_counter()
        if (step+1)%50==0 or step+1==steps:
            torch.cuda.synchronize()
            now=time.perf_counter();window_seconds.append(now-last_window);last_window=now
            if rank==0:print(f'PROGRESS backend={args.backend} step={step+1}/{steps} wall={now-started:.1f}s loss={loss.item():.4f}',flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    final_finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in raw.parameters())
    local_stats=dict(rank=rank,seconds=elapsed,first_step_seconds=first_seconds,
        post_first_seconds=elapsed-first_seconds,data_acquire_seconds=sum(waits),
        acquire_p50=percentile(waits,50),acquire_p95=percentile(waits,95),acquire_p99=percentile(waits,99),
        step_cpu_p50=percentile(latencies,50),step_cpu_p95=percentile(latencies,95),
        peak_torch_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        window_seconds=window_seconds,gradients_finite=final_finite,
        label_samples=int(seen_labels.sum()),labels_min=int(labels.min()),labels_max=int(labels.max()))
    all_stats=[None]*world
    if world>1:dist.all_gather_object(all_stats,local_stats)
    else:all_stats=[local_stats]
    if rank==0:
        slowest=max(s['seconds'] for s in all_stats)
        stable=max(s['post_first_seconds'] for s in all_stats)
        result=dict(status='complete',backend=args.backend,world=world,micro_batch=args.batch_size,
            effective_batch=world*args.batch_size,steps=steps,full_epoch=steps==full_steps,
            emitted_samples=steps*world*args.batch_size,dropped=total-steps*world*args.batch_size,
            epoch_seconds_slowest=slowest,images_per_second=steps*world*args.batch_size/slowest,
            seconds_per_update=slowest/steps,post_first_seconds_per_update=stable/(steps-1) if steps>1 else None,
            setup_seconds_rank0=setup_seconds,ranks=all_stats,manifest_sha256=data_digest,
            config=config,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            torch=torch.__version__,cuda=torch.version.cuda,
            production_source_hashes={n:hashlib.sha256((args.training_root/n).read_bytes()).hexdigest()
                for n in ['train.py','tar_data.py','models.py']},
            caveats=['original=timm original recipe; pil_matched/DALI=RRC+flip+RA(2,9)+GPU pixel erase .25',
                     'PIL/DALI operator suites, sampling order, interpolation and RNG differ; not accuracy equivalence',
                     'No cache eviction; cache/load uncontrolled; DALI memory not included in torch allocator peak',
                     'acquire time includes H2D on PIL path; timings overlap GPU work; no validation/checkpoint',
                     'Full epoch timing includes first compile/startup, post-first timing separately reported'])
        with args.result.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['ranks','config','production_source_hashes']},indent=2),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--index-dir',type=Path,required=True)
    p.add_argument('--training-root',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--backend',choices=['original','pil_matched','dali'],default='dali')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--decoder',choices=['cpu','mixed'],default='mixed')
    p.add_argument('--reader-io',choices=['mmap','plain'],default='plain')
    p.add_argument('--compile-mode',choices=['none','default'],default='default')
    p.add_argument('--epoch',type=int,default=0)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=0,help='0=one complete training epoch')
    p.add_argument('--result',type=Path)
    p.add_argument('--prepare-indexes',action='store_true')
    p.add_argument('--reader-check',action='store_true')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if min(args.batch_size,args.workers)<1 or args.steps<0:p.error('Invalid batch/workers/steps')
    if args.execute and not (args.prepare_indexes or args.reader_check) and not all([args.training_root,args.config,args.result]):
        p.error('Training benchmark requires training-root, config and new result path')
    args.root=args.root.resolve();args.index_dir=args.index_dir.resolve()
    main(args)
