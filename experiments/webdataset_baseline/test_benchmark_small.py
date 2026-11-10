"""Check batching and uint8 semantics using tiny generated JPEGs, CPU only."""
import io
import unittest

from PIL import Image
import torch

from benchmark_small import MEAN, STD, batches


class PipelineTests(unittest.TestCase):
    def test_blocking_and_output_dtype_preserve_samples_and_pixels(self):
        records = []
        for i in range(12):
            stream = io.BytesIO()
            Image.new('RGB', (24, 32), (i * 17, 83, 42)).save(stream, format='JPEG')
            records.append((stream.getvalue(), i % 2, str(i)))

        def collect(output, workers, block):
            result = {}
            for images, labels, keys in batches(records, output, workers, block, 4):
                for image, label, key in zip(images, labels, keys):
                    self.assertNotIn(key, result)
                    result[key] = (image, label)
            return result

        parent = collect('float32', 2, 1)
        block = collect('float32', 2, 4)
        compact = collect('uint8', 2, 4)
        self.assertEqual(set(parent), {str(i) for i in range(12)})
        mean, std = [torch.tensor(v).view(3, 1, 1) for v in (MEAN, STD)]
        for key in parent:
            self.assertTrue(torch.equal(parent[key][0], block[key][0]))
            self.assertEqual(parent[key][1], compact[key][1])
            converted = (compact[key][0].float()/255-mean)/std
            self.assertTrue(torch.allclose(parent[key][0], converted, atol=1e-6, rtol=0))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
