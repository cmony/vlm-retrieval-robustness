import unittest

import numpy as np

from vlmrobust.retrieval import compute_bidirectional_ranks, summarize_ranks


class RetrievalTests(unittest.TestCase):
    def test_computes_bidirectional_ranks_in_small_chunks(self):
        images = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        texts = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        caption_image_index = np.array([1, 0, 0, 1], dtype=np.int64)
        result = compute_bidirectional_ranks(images, texts, caption_image_index, chunk_size=1, device="cpu")
        np.testing.assert_array_equal(result.image_to_text, np.array([1, 1]))
        np.testing.assert_array_equal(result.text_to_image, np.array([2, 1, 2, 1]))
        summary = summarize_ranks(result.text_to_image)
        self.assertEqual(0.5, summary["r1"])
        self.assertEqual(1.0, summary["r5"])
        self.assertEqual(0.75, summary["mrr"])

    def test_rejects_caption_mapping_outside_image_range(self):
        with self.assertRaisesRegex(ValueError, "caption_image_index"):
            compute_bidirectional_ranks(
                np.eye(2, dtype=np.float32),
                np.eye(2, dtype=np.float32),
                np.array([0, 2]),
                chunk_size=2,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
