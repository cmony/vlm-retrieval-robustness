import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vlmrobust.data import load_coco_index


class CocoDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.images = root / "val2017"
        self.images.mkdir()
        images = []
        annotations = []
        ann_id = 1
        for image_id in (30, 10, 20):
            name = f"{image_id:012d}.jpg"
            Image.new("RGB", (8, 8), (image_id, 0, 0)).save(self.images / name)
            images.append({"id": image_id, "file_name": name})
            for caption_index in range(5):
                annotations.append({"id": ann_id, "image_id": image_id, "caption": f"image {image_id} caption {caption_index}"})
                ann_id += 1
        self.annotations = root / "captions.json"
        self.annotations.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")

    def test_loads_five_captions_in_annotation_order(self):
        index = load_coco_index(self.images, self.annotations, limit=3, sample_seed=42, captions_per_image=5)
        self.assertEqual(3, len(index.records))
        self.assertEqual(15, len(index.captions))
        self.assertEqual("image 10 caption 0", index.records[0].captions[0])
        self.assertEqual([0] * 5 + [1] * 5 + [2] * 5, index.caption_image_index.tolist())

    def test_seeded_subset_is_repeatable(self):
        first = load_coco_index(self.images, self.annotations, limit=2, sample_seed=7, captions_per_image=5)
        second = load_coco_index(self.images, self.annotations, limit=2, sample_seed=7, captions_per_image=5)
        self.assertEqual([x.image_id for x in first.records], [x.image_id for x in second.records])

    def test_rejects_missing_image_file(self):
        (self.images / "000000000010.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "000000000010"):
            load_coco_index(self.images, self.annotations, limit=3, sample_seed=42, captions_per_image=5)

    def test_truncates_extra_captions_deterministically(self):
        payload = json.loads(self.annotations.read_text(encoding="utf-8"))
        payload["annotations"].append(
            {"id": 999, "image_id": 10, "caption": "image 10 caption extra"}
        )
        self.annotations.write_text(json.dumps(payload), encoding="utf-8")

        index = load_coco_index(self.images, self.annotations, limit=3, sample_seed=42, captions_per_image=5)

        self.assertEqual(15, len(index.captions))
        self.assertEqual(5, len(index.records[0].captions))
        self.assertEqual(
            tuple(f"image 10 caption {caption_index}" for caption_index in range(5)),
            index.records[0].captions,
        )
        self.assertEqual((6, 7, 8, 9, 10), index.records[0].caption_ids)


if __name__ == "__main__":
    unittest.main()
