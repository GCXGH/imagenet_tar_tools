"""Synthetic USTAR/JSONL only, no full ImageNet traversal or GPU."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from epoch_benchmark import make_indexes


class IndexTests(unittest.TestCase):
    def test_offsets_sizes_names_and_reuse(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as tmp:
            root=Path(tmp);dataset=root/'data';dataset.mkdir()
            samples=[]
            with tarfile.open(dataset/'train.tar','w',format=tarfile.USTAR_FORMAT) as tar:
                for i,(size,label) in enumerate([(1,0),(512,9),(513,10),(1024,99),(2001,100),(511,999)]):
                    key=f'{i:09d}';payload=b'x'*size
                    for name,data in [(key+'.jpg',payload),(key+'.cls',str(label).encode())]:
                        member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
                    samples.append(dict(key=key,member=key+'.jpg',bytes=size,label=label))
            (dataset/'meta.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
            m=dict(format='imagenet-original-webdataset-v1',complete=True,limit_per_class=0,
                   splits=dict(train=dict(shards=[dict(path='train.tar',metadata='meta.jsonl',samples=len(samples),
                                                       bytes=(dataset/'train.tar').stat().st_size)])))
            (dataset/'manifest.json').write_text(json.dumps(m))
            make_indexes(dataset,root/'index');make_indexes(dataset,root/'index')
            index=(root/'index/train.idx').read_text().splitlines()
            self.assertEqual(index[0],'v1.2 6')
            with tarfile.open(dataset/'train.tar') as tar, (dataset/'train.tar').open('rb') as stream:
                members=list(tar)
                for i,line in enumerate(index[1:]):
                    fields=line.split()
                    self.assertEqual(len(fields),8)
                    for component in range(2):
                        ext,offset,size,name=fields[component*4:component*4+4]
                        member=members[i*2+component]
                        self.assertEqual((int(offset),int(size),name),(member.offset_data,member.size,member.name))
                        stream.seek(int(offset));data=stream.read(int(size))
                        if component:self.assertEqual(int(data),samples[i]['label'])
            (dataset/'manifest.json').write_text(json.dumps(dict(m,changed=True)))
            with self.assertRaises(ValueError):make_indexes(dataset,root/'index')


if __name__=='__main__':unittest.main()
