import math
import unittest

import numpy as np

from spinevision_clinical_metrics import (
    compute_canal_mask_measurement,
    compute_clinical_metrics,
    compute_l1_s1_lordosis,
    compute_l4_s1_lordosis,
    compute_sacral_slope,
    infer_dicom_slice_plane,
    infer_nifti_slice_plane,
)


def _sagittal_nifti_meta():
    # array axis 0 -> RAS anterior, axis 1 -> RAS superior,
    # array axis 2 -> RAS right/left (therefore sagittal slices)
    affine = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return {
        "source_format": "nifti",
        "slice_plane": "sagittal",
        "nifti_affine": affine.tolist(),
        "spacing_values_mm": [1.0, 1.0, 1.0],
        "label_semantics_confirmed": True,
    }


def _vertebral_strip(label_map, label_id, superior_line, thickness=8):
    for ap_index in range(20, 81):
        superior = int(round(superior_line(ap_index)))
        label_map[ap_index, superior - thickness : superior + 1] = label_id


def _raster_vertebral_strip(label_map, label_id, superior_row, thickness=8):
    for column in range(20, 81):
        row = int(round(superior_row(column)))
        label_map[row : row + thickness + 1, column] = label_id


class ClinicalMetricTests(unittest.TestCase):
    def test_plane_inference(self):
        meta = _sagittal_nifti_meta()
        self.assertEqual(infer_nifti_slice_plane(meta["nifti_affine"]), "sagittal")
        # DICOM row direction posterior/anterior, column direction superior/inferior.
        self.assertEqual(
            infer_dicom_slice_plane([0, 1, 0, 0, 0, 1]),
            "sagittal",
        )

    def test_l1_s1_lordosis_uses_endplate_angle(self):
        label_map = np.zeros((110, 130), dtype=np.uint8)
        _vertebral_strip(label_map, 1, lambda _ap: 100.0)
        slope = math.tan(math.radians(30.0))
        _vertebral_strip(label_map, 6, lambda ap: 25.0 + slope * (ap - 20.0))

        result = compute_l1_s1_lordosis([label_map], _sagittal_nifti_meta())

        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["numeric_value"], 30.0, delta=2.0)
        self.assertTrue(result["requires_human_review"])
        self.assertIn("不自动判定", result["method"])

    def test_l4_s1_lordosis_and_sacral_slope(self):
        label_map = np.zeros((110, 130), dtype=np.uint8)
        l4_slope = math.tan(math.radians(10.0))
        s1_slope = math.tan(math.radians(30.0))
        _vertebral_strip(label_map, 4, lambda ap: 75.0 + l4_slope * (ap - 20.0))
        _vertebral_strip(label_map, 6, lambda ap: 25.0 + s1_slope * (ap - 20.0))

        lower_lordosis = compute_l4_s1_lordosis(
            [label_map], _sagittal_nifti_meta()
        )
        sacral_slope = compute_sacral_slope([label_map], _sagittal_nifti_meta())

        self.assertEqual(lower_lordosis["status"], "measured")
        self.assertAlmostEqual(lower_lordosis["numeric_value"], 20.0, delta=2.0)
        self.assertEqual(sacral_slope["status"], "measured")
        self.assertAlmostEqual(sacral_slope["numeric_value"], 30.0, delta=2.0)
        self.assertIn("不套用站立位正常范围", sacral_slope["method"])

    def test_raster_allows_rotation_invariant_lordosis_only(self):
        label_map = np.zeros((130, 110), dtype=np.uint8)
        _raster_vertebral_strip(label_map, 1, lambda _x: 15.0)
        slope = math.tan(math.radians(30.0))
        _raster_vertebral_strip(
            label_map, 6, lambda x: 95.0 - slope * (x - 20.0)
        )
        meta = {
            "source_format": "raster",
            "slice_plane": "sagittal",
            "label_semantics_confirmed": True,
            "measurement_geometry": "pixel_assumed_square",
            "raster_patient_horizontal_confirmed": False,
        }

        lordosis = compute_l1_s1_lordosis([label_map], meta)
        sacral_slope = compute_sacral_slope([label_map], meta)

        self.assertEqual(lordosis["status"], "measured")
        self.assertAlmostEqual(lordosis["numeric_value"], 30.0, delta=2.0)
        self.assertIn("方形像素假设", lordosis["value"])
        self.assertEqual(sacral_slope["status"], "unavailable")
        self.assertIn("患者水平轴", sacral_slope["value"])

    def test_lordosis_rejects_unknown_plane(self):
        result = compute_l1_s1_lordosis(
            [np.zeros((64, 64), dtype=np.uint8)],
            {
                "source_format": "png",
                "slice_plane": "unknown",
                "label_semantics_confirmed": True,
            },
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("矢状位", result["value"])

    def test_lordosis_rejects_unconfirmed_label_dictionary(self):
        result = compute_l1_s1_lordosis(
            [np.zeros((64, 64), dtype=np.uint8)],
            {"source_format": "nifti", "slice_plane": "sagittal"},
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("正式标签字典", result["value"])

    def test_canal_volume_is_descriptive_only(self):
        label_map = np.zeros((20, 20), dtype=np.uint8)
        label_map[:10, :10] = 11
        label_map[10:15, :10] = 12
        result = compute_canal_mask_measurement(
            [label_map],
            {"spacing_values_mm": [1.0, 2.0, 3.0]},
        )
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["numeric_value"], 0.9)
        self.assertIn("不得解释为狭窄分级", result["method"])

    def test_unsupported_diagnoses_fail_closed(self):
        rows = compute_clinical_metrics(
            [np.zeros((64, 64), dtype=np.uint8)],
            {"source_format": "png", "slice_plane": "unknown"},
        )
        by_key = {row["key"]: row for row in rows}
        for key in (
            "椎间盘高度/退变",
            "椎管狭窄分级",
            "骨密度/骨质疏松",
            "椎体压缩性骨折",
        ):
            self.assertEqual(by_key[key]["status"], "unavailable")
            self.assertTrue(by_key[key]["warn"])


if __name__ == "__main__":
    unittest.main()
