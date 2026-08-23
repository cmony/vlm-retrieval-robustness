import sys
import types
import unittest
from unittest.mock import patch

from vlmrobust.models import HuggingFaceAdapter, SiglipAdapter, load_model_adapter


class ModelAdapterTests(unittest.TestCase):
    def test_fp32_cuda_does_not_enable_half_precision_autocast(self):
        adapter = HuggingFaceAdapter(model=None, processor=None, device="cuda", precision="fp32")
        with adapter._autocast():
            value = 1
        self.assertEqual(1, value)

    def test_siglip_uses_fixed_length_text_padding(self):
        self.assertEqual("max_length", SiglipAdapter.text_padding)

    def test_load_model_adapter_uses_dtype_not_torch_dtype(self):
        class FakeProcessor:
            called_with = None

            @classmethod
            def from_pretrained(cls, checkpoint, **kwargs):
                cls.called_with = (checkpoint, kwargs)
                return cls()

        class FakeModel:
            called_with = None

            @classmethod
            def from_pretrained(cls, checkpoint, **kwargs):
                cls.called_with = (checkpoint, kwargs)
                return cls()

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                return self

        fake_torch = types.SimpleNamespace(
            bfloat16="bf16-dtype",
            float16="fp16-dtype",
            float32="fp32-dtype",
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                is_bf16_supported=lambda: True,
            ),
        )
        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            BlipForImageTextRetrieval=FakeModel,
            CLIPModel=FakeModel,
            SiglipModel=FakeModel,
        )

        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            load_model_adapter(
                {"name": "clip_vit_b32", "type": "clip", "checkpoint": "openai/clip-vit-base-patch32"},
                device="cuda",
                precision="bf16",
            )

        _, model_kwargs = FakeModel.called_with
        self.assertEqual("bf16-dtype", model_kwargs["dtype"])
        self.assertNotIn("torch_dtype", model_kwargs)


if __name__ == "__main__":
    unittest.main()
