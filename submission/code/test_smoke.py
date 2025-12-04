import os
from pathlib import Path

import numpy as np


def test_paths_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "data" / "images").exists()
    assert (root / "models").exists()
    assert (root / "results").exists()


def test_mask_blend_is_bounded():
    # Ensure the blending logic keeps values within [0,255]
    res_np = np.full((2, 2, 3), 255, dtype=np.float32)
    orig_np = np.zeros((2, 2, 3), dtype=np.float32)
    mask_arr = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    mask_3ch = np.stack([mask_arr] * 3, axis=-1)
    final_np = res_np * mask_3ch + orig_np * (1 - mask_3ch)
    assert final_np.min() >= 0.0
    assert final_np.max() <= 255.0
