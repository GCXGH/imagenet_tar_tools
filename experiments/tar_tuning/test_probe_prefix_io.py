"""Tiny synthetic archive test; no ImageNet reads."""
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from probe_prefix_io import measure, window


class PrefixProbeTest(unittest.TestCase):
    def test_same_samples_and_less_prefix_reading(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as tmp:
            root = Path(tmp)
            records = []
            with tarfile.open(root / 'train.tar', 'w', format=tarfile.USTAR_FORMAT) as archive:
                for i in range(48):
                    payload = bytes([i]) * 200000
                    key = f'{i:09d}'
                    for name, data in ((key + '.jpg', payload), (key + '.cls', b'0')):
                        header = tarfile.TarInfo(name)
                        header.size = len(data)
                        archive.addfile(header, io.BytesIO(data))
                    records.append(dict(key=key, member=key + '.jpg', bytes=len(payload),
                                        label=0, sha256=hashlib.sha256(payload).hexdigest()))
            (root / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            (root / 'manifest.json').write_text(json.dumps(dict(
                format='imagenet-original-webdataset-v1', complete=True,
                splits=dict(train=dict(shards=[dict(path='train.tar', metadata='train.jsonl', samples=48)])))))
            path, samples = window(root, 0, 40, 4)
            budget = dict(remaining=64 * 1024**2)
            baseline = measure(path, samples, 40, False, budget)
            indexed = measure(path, samples, 40, True, budget)
            self.assertEqual(baseline['sample_payload_order_sha256'], indexed['sample_payload_order_sha256'])
            self.assertLess(indexed['raw_read_bytes'], baseline['raw_read_bytes'])
            with self.assertRaises(ValueError):
                window(root, 0, 47, 2)
            with self.assertRaises(RuntimeError):
                measure(path, samples, 40, False, dict(remaining=1))


if __name__ == '__main__':
    unittest.main()
