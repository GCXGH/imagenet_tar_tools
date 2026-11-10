"""Tiny generated image fixtures only, no real ImageNet traversal or GPU."""
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torchvision.transforms import ToTensor

from loader import EpochLoader, encoded_stream, plan_epoch, unpack_sample


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp/opencode')
        self.root = Path(self.tmp.name)
        shards, self.expected = [], {}
        for index, count in enumerate((7, 3, 8, 4, 6, 5, 2)):
            path = self.root / f'{index}.tar'
            with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as archive:
                for i in range(count):
                    key = f'{index:03d}{i:03d}'
                    stream = io.BytesIO()
                    Image.new('RGB', (12, 10), (index * 20, i * 10, 0)).save(stream, format='JPEG')
                    payload, label = stream.getvalue(), i % 2
                    self.expected[key] = (label, hashlib.sha256(payload).hexdigest())
                    for name, data in ((key + '.jpg', payload), (key + '.cls', str(label).encode())):
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
            shards.append(dict(path=path.name, samples=count))
        self.manifest = dict(complete=True, class_to_idx=dict(a=0, b=1),
                             splits={s: dict(samples=35, shards=shards) for s in ('train', 'val')})
        self.write_manifest()

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def keys(self, **kwargs):
        loader = EpochLoader(self.root, 'train', ToTensor(), 4, **kwargs)
        keys = []
        for images, labels, identifiers in loader:
            self.assertEqual(tuple(images.shape[1:]), (3, 10, 12))
            for label, key in zip(labels.tolist(), identifiers):
                self.assertEqual(label, self.expected[key][0])
            keys.extend(identifiers)
        return loader, keys

    def test_original_bytes_and_labels(self):
        for sample in encoded_stream([str(self.root / '0.tar')]):
            payload, label, key = unpack_sample(sample, 2)
            self.assertEqual((label, hashlib.sha256(payload).hexdigest()), self.expected[key])

    def test_coverage_disjoint_ranks_and_workers(self):
        for workers in (0, 2):
            groups = [self.keys(world=2, rank=r, workers=workers)[1] for r in range(2)]
            self.assertFalse(set(groups[0]) & set(groups[1]))
            self.assertEqual(set(groups[0] + groups[1]), set(self.expected))
            self.assertEqual(sum(map(len, groups)), 35)

    def test_equal_steps_no_worker_tail_loss(self):
        for workers in (0, 2):
            outputs = [self.keys(world=2, rank=r, workers=workers, policy='equal_steps') for r in range(2)]
            self.assertEqual(len(outputs[0][0]), len(outputs[1][0]))
            self.assertEqual(len(outputs[0][1]), len(outputs[1][1]))
            self.assertFalse(set(outputs[0][1]) & set(outputs[1][1]))
            self.assertEqual(35 - sum(len(keys) for _, keys in outputs), outputs[0][0].plan['dropped_total'])

    def test_epoch_reproducibility(self):
        a = self.keys(workers=2, seed=3, epoch=4)[1]
        self.assertEqual(a, self.keys(workers=2, seed=3, epoch=4)[1])
        self.assertNotEqual(a, self.keys(workers=2, seed=3, epoch=5)[1])

    def test_validation_empty_rank_and_workers(self):
        loader = EpochLoader(self.root, 'val', ToTensor(), 4, world=10, rank=9, workers=2)
        self.assertEqual(list(loader), [])

    def test_bad_counts_and_labels_raise(self):
        with self.assertRaises(ValueError):
            unpack_sample(dict(jpg=b'x', cls=b'2', __key__='bad'), 2)
        self.manifest['splits']['train']['shards'][0]['samples'] += 1
        self.manifest['splits']['train']['samples'] += 1
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.keys(shuffle_buffer=0)

    def test_plan_rejects_insufficient_training_shards(self):
        with self.assertRaises(ValueError):
            plan_epoch(self.manifest['splits']['train']['shards'], 10, 4, policy='equal_steps')


if __name__ == '__main__':
    unittest.main()
