#!/usr/bin/env python3
"""Sequential full verification of shards, labels, IDs and original byte hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from prepare import FORMAT, sha256_file


def verify(root, check_source=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['format'] == FORMAT and manifest['complete']
    assert json.loads((root / 'class_to_idx.json').read_text()) == manifest['class_to_idx']
    for split, info in manifest['splits'].items():
        count = 0
        seen = set()
        sources = set()
        labels = set()
        for shard in info['shards']:
            path = root / shard['path']
            metadata = root / shard['metadata']
            assert path.stat().st_size == shard['bytes']
            assert sha256_file(path) == shard['sha256'], path
            assert sha256_file(metadata) == shard['metadata_sha256'], metadata
            local = 0
            with tarfile.open(path, 'r|') as archive, metadata.open() as stream:
                iterator = iter(archive)
                for line in stream:
                    sample = json.loads(line)
                    image_member = next(iterator)
                    assert image_member.isfile()
                    assert image_member.name == sample['member']
                    data = archive.extractfile(image_member).read()
                    label_member = next(iterator)
                    assert label_member.isfile()
                    assert label_member.name == sample['key'] + '.cls'
                    label = int(archive.extractfile(label_member).read())
                    assert len(data) == sample['bytes']
                    assert hashlib.sha256(data).hexdigest() == sample['sha256']
                    assert label == sample['label']
                    assert 0 <= label < len(manifest['class_to_idx'])
                    assert sample['key'] not in seen and sample['source'] not in sources
                    assert manifest['class_to_idx'][sample['source'].split('/')[0]] == label
                    if check_source:
                        original = Path(manifest['source']) / split / sample['source']
                        assert sha256_file(original) == sample['sha256']
                    seen.add(sample['key'])
                    sources.add(sample['source'])
                    labels.add(label)
                    local += 1
                assert next(iterator, None) is None
            assert local == shard['samples']
            count += local
            print(f'verified {shard["path"]}: {local}', flush=True)
        assert count == info['samples']
        assert len(labels) == len(manifest['class_to_idx'])
        print(f'{split}: verified {count} unique samples, {len(labels)} classes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--check-source', action='store_true',
                        help='Also reread original files; archive hashes are always checked')
    args = parser.parse_args()
    verify(args.root, args.check_source)
