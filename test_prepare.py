"""Standard-library conversion regressions; never touches the real ImageNet."""
from pathlib import Path
import tempfile
import unittest

from prepare import convert
from verify import verify


class ConversionTests(unittest.TestCase):
    def test_bytes_resume_determinism_and_rejections(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode', prefix='converter-test-') as tmp:
            root = Path(tmp)
            for split in ('train', 'val'):
                for label in ('a', 'b'):
                    directory = root / 'source' / split / label
                    directory.mkdir(parents=True)
                    for i in range(7):
                        (directory / f'{i}.jpg').write_bytes(f'{split}-{label}-{i}'.encode())
            manifest = convert(root / 'source', root / 'tar', 5, 0, 2)
            verify(root / 'tar', check_source=True)
            self.assertEqual(convert(root / 'source', root / 'tar', 5, 0, 2), manifest)
            other = convert(root / 'source', root / 'tar2', 5, 0, 2)
            self.assertEqual(manifest, other)
            with self.assertRaises(ValueError):
                convert(root / 'source', root / 'tar', 5, 1, 2)
            with self.assertRaises(ValueError):
                convert(root / 'source', root / 'source' / 'nested', 5, 0, 2)
            shard = root / 'tar' / manifest['splits']['train']['shards'][0]['path']
            data = bytearray(shard.read_bytes())
            data[512] ^= 1
            shard.write_bytes(data)
            with self.assertRaises(AssertionError):
                verify(root / 'tar')
            with self.assertRaises(ValueError):
                convert(root / 'source', root / 'tar', 5, 0, 2)


if __name__ == '__main__':
    unittest.main()
