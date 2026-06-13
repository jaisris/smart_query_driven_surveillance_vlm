"""Tests for CLIPEncoder output shape and L2 normalisation.

Does NOT load actual CLIP weights — patches the HuggingFace model/processor
so tests run fast with no internet or GPU.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def _make_mock_encoder():
    """Return a CLIPEncoder with mocked HuggingFace internals."""
    with patch("models.clip_encoder.CLIPModel") as MockModel, \
         patch("models.clip_encoder.CLIPProcessor") as MockProcessor:

        # Processor returns tensors of the right shapes
        mock_proc = MagicMock()
        mock_proc.return_value = {
            "pixel_values": torch.zeros(1, 3, 224, 224),
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        MockProcessor.from_pretrained.return_value = mock_proc

        # Model returns random (but deterministic) 512-d features
        mock_model = MagicMock()
        mock_model.get_image_features.return_value = torch.randn(1, 512)
        mock_model.get_text_features.return_value = torch.randn(1, 512)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        MockModel.from_pretrained.return_value = mock_model

        from models.clip_encoder import CLIPEncoder
        from utils.config_loader import AppConfig
        return CLIPEncoder(config=AppConfig())


def test_encode_image_shape():
    enc = _make_mock_encoder()
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    vec = enc.encode_image(frame)
    assert vec.shape == (512,)


def test_encode_image_l2_normalised():
    enc = _make_mock_encoder()
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    vec = enc.encode_image(frame)
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_encode_text_shape():
    enc = _make_mock_encoder()
    vec = enc.encode_text("person running near entrance")
    assert vec.shape == (512,)


def test_encode_text_l2_normalised():
    enc = _make_mock_encoder()
    vec = enc.encode_text("loitering individual")
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_encode_image_batch_shape():
    enc = _make_mock_encoder()
    # Patch to return batch output
    enc.model.get_image_features.return_value = torch.randn(4, 512)
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
    mat = enc.encode_image_batch(frames)
    assert mat.shape == (4, 512)
    # Each row should be L2-normalised
    norms = np.linalg.norm(mat, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)
