#!/usr/bin/env python3
"""Explicitly authorized three-case suite, serial only, timeout per process group.

Default dry-run. Outputs logs, GPU/system telemetry and JSON, never checkpoints.
No downloads, conversion, cache eviction, package changes, or automatic retries.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def telemetry():
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
        gpu=gpu.stdout.strip().splitlines()
    except subprocess.TimeoutExpired:
        gpu=['telemetry timeout']
    mem={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
         if k in ('MemAvailable','MemFree','Cached','SwapFree')}
    disks=[line.split() for line in Path('/proc/diskstats').read_text().splitlines() if line.split()[2] in ('sda','sdb')]
    return dict(unix=time.time(),gpu_csv=gpu,mem_kib=mem,diskstats=disks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--suite',choices=['baseline','decode_ablation'],default='baseline')
    args=p.parse_args()
    here=Path(__file__).resolve().parent
    common=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
        str(here/'epoch_benchmark.py'),
        '--root','/data1/xinghaitong/dataset/imagenet1k_original_tar_v1',
        '--index-dir','/data1/xinghaitong/dataset/imagenet_dali_indexes_v1_2',
        '--training-root','/data1/xinghaitong/projects/spb_vit_dist/ablation/pbi_width_scaling',
        '--config','/data1/xinghaitong/projects/spb_vit_dist/ablation/pbi_width_scaling/configs_4/pbi_wide_100e.yaml',
        '--batch-size','256','--workers','8','--compile-mode','default','--epoch','0','--execute']
    cases=([('original','original','mixed'),('pil_matched','pil_matched','mixed'),('dali','dali','mixed')]
           if args.suite=='baseline' else
           [('dali_mixed','dali','mixed'),('dali_cpu','dali','cpu'),('original_reverse','original','mixed')])
    if not args.execute:
        print(json.dumps(dict(cases=cases,GPUs='4,5,6,7',global_batch=1024,steps_per_case=1251,
                              timeout_per_case=1800,output=str(args.output),common_command=common),indent=2));return
    # Before creating processes, assert all selected physical GPUs remain free.
    check=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True,timeout=10)
    for line in check.stdout.splitlines():
        index,memory=[int(v.strip()) for v in line.split(',')]
        if index in (4,5,6,7) and memory>100:
            raise RuntimeError(f'GPU{index} already allocated ({memory}MiB); refusing to start')
    args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='1',
             TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR='/tmp/opencode/width_compile_cache')
    (args.output/'suite.json').write_text(json.dumps(dict(cases=cases,command=common,
        environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','TORCHINDUCTOR_CACHE_DIR')},
        started=datetime.datetime.now().isoformat(),timeout_per_case=1800),indent=2)+'\n')
    summaries=[]
    for name,backend,decoder in cases:
        print('START',name,datetime.datetime.now().isoformat(),flush=True)
        command=common+['--backend',backend,'--decoder',decoder,'--result',str(args.output/(name+'.json'))]
        started=time.monotonic()
        with (args.output/(name+'.log')).open('x') as log, (args.output/(name+'_telemetry.jsonl')).open('x') as monitor:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:
                while child.poll() is None:
                    monitor.write(json.dumps(telemetry())+'\n');monitor.flush()
                    if time.monotonic()-started>1800:
                        raise TimeoutError('Per-case 30-minute limit reached')
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f'{name} failed with code {child.returncode}; see {log.name}; suite stops')
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        result=json.loads((args.output/(name+'.json')).read_text())
        summary={k:result[k] for k in ('backend','steps','effective_batch','images_per_second',
                    'epoch_seconds_slowest','post_first_seconds_per_update')}
        summary.update(case=name,decoder=decoder)
        summaries.append(summary)
        (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print('DONE',json.dumps(summary),flush=True)
    print('SUITE COMPLETE',str(args.output),flush=True)


if __name__=='__main__':main()
