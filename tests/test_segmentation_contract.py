from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "SpineVision_AI_real_infer.py"
SPEC = importlib.util.spec_from_file_location("spinevision_gui", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载模块: {MODULE_PATH}")
GUI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUI)


class SegmentationContractTests(unittest.TestCase):
    def test_background_plus_twelve_foreground_classes(self):
        self.assertEqual(GUI.BACKGROUND_LABEL, 0)
        self.assertEqual(GUI.FOREGROUND_CLASS_COUNT, 12)
        self.assertEqual(GUI.MODEL_TOTAL_CLASSES, 13)
        self.assertEqual(len(GUI.SEG_CLASSES), 12)
        self.assertEqual(len(GUI.SEG_COLORS_RGB), 12)
        self.assertNotIn("硬膜囊", [row[0] for row in GUI.SEG_CLASSES])

    def test_best_checkpoint_is_default_baseline(self):
        with mock.patch.object(GUI, "DEFAULT_WEIGHT_PATH", ""):
            resolved = Path(GUI.resolve_default_model_path()).resolve()
        self.assertEqual(resolved, (ROOT / "runslite_256" / "best.pt").resolve())

    def test_thirteen_channel_logits_map_to_labels_zero_through_twelve(self):
        worker = GUI.SegWorker([], model_path=str(ROOT / "runslite_256" / "best.pt"))
        self.assertEqual(worker.num_classes, 13)

        logits = torch.full((1, 13, 2, 2), -10.0)
        expected = np.array([[0, 1], [11, 12]], dtype=np.uint8)
        for y in range(2):
            for x in range(2):
                logits[0, int(expected[y, x]), y, x] = 10.0

        restore = {
            "orig_hw": (2, 2),
            "src": (0, 2, 0, 2),
            "dst": (0, 2, 0, 2),
        }
        _, label_map = worker._logits_to_masks(logits, restore, torch, torch.nn.functional)
        np.testing.assert_array_equal(label_map, expected)
        self.assertLessEqual(int(label_map.max()), 12)

    def test_input_domain_labels(self):
        self.assertEqual(GUI.TRAINED_INPUT_DOMAIN, "训练输入格式")
        self.assertEqual(GUI.COMPATIBLE_INPUT_DOMAIN, "兼容输入（待验证）")
        self.assertTrue(GUI.LABEL_SEMANTICS_CONFIRMED)


if __name__ == "__main__":
    unittest.main()
