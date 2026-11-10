#!/usr/bin/env python3
"""Read-only summarizer for completed cases; no benchmark processes launched."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(root):
    rows=[]
    cases=[p.stem for p in root.glob('*.json') if p.stem not in ('suite','summary')]
    for case in sorted(cases):
        path=root/(case+'.json')
        if not path.exists():continue
        result=json.loads(path.read_text())
        telemetry=[json.loads(line) for line in (root/(case+'_telemetry.jsonl')).read_text().splitlines()]
        gpus=[]
        for sample in telemetry:
            for line in sample['gpu_csv']:
                try:fields=[float(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if int(fields[0]) in (4,5,6,7):gpus.append(fields)
        disk=[]
        for sample in telemetry:
            for entry in sample['diskstats']:
                if entry[2]=='sdb':disk.append((sample['unix'],entry))
        storage={}
        if len(disk)>1:
            start,a=disk[0];end,b=disk[-1];seconds=end-start
            sectors=int(b[5])-int(a[5]);io_ms=int(b[12])-int(a[12])
            reads=int(b[3])-int(a[3]);read_ms=int(b[6])-int(a[6])
            storage=dict(read_gib=sectors*512/1024**3,read_mib_per_second=sectors*512/1024**2/seconds,
                         busy_pct=io_ms/seconds/10,read_await_ms=read_ms/reads if reads else 0,
                         caveat='whole-device shared workload, includes startup/compile; not process-attributed')
        rows.append(dict(case=case,backend=result['backend'],decoder=result['settings']['decoder'],
            epoch_seconds=result['epoch_seconds_slowest'],
            seconds_per_update=result['seconds_per_update'],post_first=result['post_first_seconds_per_update'],
            images_per_second=result['images_per_second'],steps=result['steps'],
            acquire_seconds_by_rank=[r['data_acquire_seconds'] for r in result['ranks']],
            gpu_util_mean=statistics.mean(g[2] for g in gpus) if gpus else None,
            total_gpu_memory_peak_gib=max(g[1] for g in gpus)/1024 if gpus else None,
            mem_available_min_gib=min(t['mem_kib']['MemAvailable'] for t in telemetry)/1024**2 if telemetry else None,
            storage=storage))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    args=p.parse_args();print(json.dumps(analyze(args.root),indent=2))
