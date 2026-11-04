"""Evidence-bounded clinical measurements for SpineVision.

This module deliberately separates:

1. measurements that can be derived from the current masks and image geometry;
2. clinical classifications that require additional image content or landmarks.

It must not convert display intensity or an unvalidated mask ratio into a
diagnosis.  Every returned item therefore includes a status and an explicit
input limitation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


CLINICAL_REFERENCES = {
    "lordosis_l1_s1": {
        "method": "L1–S1 Cobb lumbar lordosis",
        "definition": "Angle between the superior endplates of L1 and S1.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/37600414/",
    },
    "lordosis_l4_s1": {
        "method": "L4–S1 lower lumbar lordosis",
        "definition": "Angle between the superior endplates of L4 and S1.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6259763/",
    },
    "sacral_slope": {
        "method": "Sacral slope",
        "definition": "Angle between the superior endplate of S1 and the horizontal line.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9993244/",
    },
    "stenosis_schizas": {
        "method": "Schizas qualitative lumbar stenosis grading",
        "definition": "Axial T2 dural-sac/rootlet/CSF morphology; not an aggregate mask ratio.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/20671589/",
    },
    "stenosis_wfns": {
        "method": "WFNS lumbar spinal stenosis recommendations",
        "definition": "No single unambiguous quantitative imaging criterion; imaging and symptoms may disagree.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7322797/",
    },
    "ct_bmd": {
        "method": "Opportunistic CT L1 trabecular attenuation",
        "definition": "Requires calibrated CT Hounsfield units and a trabecular L1 ROI.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/23588747/",
    },
    "genant": {
        "method": "Genant semiquantitative vertebral fracture assessment",
        "definition": "Requires anterior/middle/posterior vertebral-height assessment plus visual differentiation of non-fracture deformity.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/8237484/",
    },
    "disc_height": {
        "method": "MRI area-based intervertebral disc height",
        "definition": "Requires a disc contour or validated vertebral endplate landmarks.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/29432200/",
    },
}


def infer_nifti_slice_plane(affine: Sequence[Sequence[float]]) -> str:
    """Infer the plane of array slices ``data[:, :, z]`` from a NIfTI affine."""
    try:
        matrix = np.asarray(affine, dtype=np.float64)
        normal = matrix[:3, 2]
        axis = int(np.argmax(np.abs(normal)))
    except Exception:
        return "unknown"
    return ("sagittal", "coronal", "axial")[axis]


def infer_dicom_slice_plane(image_orientation: Sequence[float]) -> str:
    """Infer DICOM slice plane from ImageOrientationPatient direction cosines."""
    try:
        orientation = np.asarray(image_orientation, dtype=np.float64).reshape(2, 3)
        normal = np.cross(orientation[0], orientation[1])
        axis = int(np.argmax(np.abs(normal)))
    except Exception:
        return "unknown"
    # DICOM patient coordinates are LPS: X left/right, Y anterior/posterior,
    # Z inferior/superior. The dominant normal therefore identifies the plane.
    return ("sagittal", "coronal", "axial")[axis]


def _metric(
    key: str,
    value: str,
    *,
    status: str,
    warn: bool,
    method: str,
    reference_ids: Iterable[str] = (),
    numeric_value: Optional[float] = None,
    unit: str = "",
) -> Dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "status": status,
        "warn": warn,
        "method": method,
        "reference_ids": list(reference_ids),
        "numeric_value": numeric_value,
        "unit": unit,
        "requires_human_review": True,
    }


def _unavailable(
    key: str,
    reason: str,
    *,
    method: str,
    reference_ids: Iterable[str],
) -> Dict[str, Any]:
    return _metric(
        key,
        f"不可计算：{reason}",
        status="unavailable",
        warn=True,
        method=method,
        reference_ids=reference_ids,
    )


def _representative_slice(
    label_maps: Sequence[np.ndarray], first_label: int, second_label: int
) -> Optional[int]:
    best_index = None
    best_score = 0
    for index, label_map in enumerate(label_maps):
        first_count = int(np.count_nonzero(label_map == first_label))
        second_count = int(np.count_nonzero(label_map == second_label))
        score = min(first_count, second_count)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= 30 else None


def _representative_single_label_slice(
    label_maps: Sequence[np.ndarray], label_id: int
) -> Optional[int]:
    best_index = None
    best_count = 0
    for index, label_map in enumerate(label_maps):
        count = int(np.count_nonzero(label_map == label_id))
        if count > best_count:
            best_count = count
            best_index = index
    return best_index if best_count >= 30 else None


def _nifti_ap_si_coordinates(
    rows: np.ndarray,
    columns: np.ndarray,
    slice_index: int,
    meta: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        affine = np.asarray(meta["nifti_affine"], dtype=np.float64)
        voxels = np.column_stack(
            [
                rows.astype(np.float64),
                columns.astype(np.float64),
                np.full(rows.shape, float(slice_index)),
                np.ones(rows.shape, dtype=np.float64),
            ]
        )
        ras = voxels @ affine.T
        # RAS Y is anterior/posterior and RAS Z is inferior/superior.
        return ras[:, 1], ras[:, 2]
    except Exception:
        return None


def _dicom_ap_si_coordinates(
    rows: np.ndarray,
    columns: np.ndarray,
    slice_index: int,
    meta: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        orientation = np.asarray(meta["dicom_image_orientation"], dtype=np.float64)
        row_direction = orientation[:3]
        column_direction = orientation[3:6]
        spacing = np.asarray(meta["dicom_pixel_spacing"], dtype=np.float64)
        positions = meta.get("dicom_image_positions") or [[0.0, 0.0, 0.0]]
        position = np.asarray(positions[min(slice_index, len(positions) - 1)], dtype=np.float64)
        points = (
            position[None, :]
            + columns[:, None] * float(spacing[1]) * row_direction[None, :]
            + rows[:, None] * float(spacing[0]) * column_direction[None, :]
        )
        # DICOM LPS Y is anterior/posterior and Z is inferior/superior.
        return points[:, 1], points[:, 2]
    except Exception:
        return None


def _physical_ap_si_coordinates(
    rows: np.ndarray,
    columns: np.ndarray,
    slice_index: int,
    meta: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    source_format = str(meta.get("source_format", "")).lower()
    if source_format == "nifti":
        return _nifti_ap_si_coordinates(rows, columns, slice_index, meta)
    if source_format == "dicom":
        return _dicom_ap_si_coordinates(rows, columns, slice_index, meta)
    if source_format == "raster":
        # For a visibly upright sagittal raster image, columns represent the
        # anterior-posterior image axis and negative rows the superior axis.
        # A common scalar pixel size cancels in an angle between two lines.
        return columns.astype(np.float64), -rows.astype(np.float64)
    return None


def _fit_superior_endplate(
    label_map: np.ndarray,
    label_id: int,
    slice_index: int,
    meta: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, float]]:
    rows, columns = np.where(label_map == label_id)
    if rows.size < 30:
        return None
    coordinates = _physical_ap_si_coordinates(rows, columns, slice_index, meta)
    if coordinates is None:
        return None
    ap, si = coordinates
    if not np.all(np.isfinite(ap)) or not np.all(np.isfinite(si)):
        return None

    ap_min, ap_max = float(np.min(ap)), float(np.max(ap))
    ap_span = ap_max - ap_min
    if ap_span < 5.0:
        return None

    # Extract a robust superior envelope. Central bins avoid rounded vertebral
    # corners; the 95th percentile limits isolated mask noise.
    bin_count = int(np.clip(np.sqrt(rows.size), 12, 36))
    edges = np.linspace(ap_min, ap_max, bin_count + 1)
    envelope_ap: List[float] = []
    envelope_si: List[float] = []
    for index in range(bin_count):
        in_bin = (ap >= edges[index]) & (
            ap <= edges[index + 1] if index == bin_count - 1 else ap < edges[index + 1]
        )
        if int(np.count_nonzero(in_bin)) < 2:
            continue
        envelope_ap.append(float(np.median(ap[in_bin])))
        envelope_si.append(float(np.percentile(si[in_bin], 95)))

    if len(envelope_ap) < 8:
        return None
    x = np.asarray(envelope_ap, dtype=np.float64)
    y = np.asarray(envelope_si, dtype=np.float64)
    low, high = np.quantile(x, [0.15, 0.85])
    keep = (x >= low) & (x <= high)
    x, y = x[keep], y[keep]
    if x.size < 6 or float(np.ptp(x)) < 4.0:
        return None

    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (slope * x + intercept)
    rmse_mm = float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(slope) or rmse_mm > 4.0:
        return None
    direction = np.asarray([1.0, float(slope)], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return direction, rmse_mm


def compute_l1_s1_lordosis(
    label_maps: Sequence[np.ndarray],
    meta: Optional[Dict[str, Any]],
    *,
    l1_label: int = 1,
    s1_label: int = 6,
) -> Dict[str, Any]:
    """Measure L1–S1 lordosis without applying a diagnostic normality cutoff."""
    meta = dict(meta or {})
    if not bool(meta.get("label_semantics_confirmed", False)):
        return _unavailable(
            "L1–S1 腰椎前凸角",
            "需先确认 label 1=L1、label 6=S1 的正式标签字典",
            method="L1 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l1_s1",),
        )
    if meta.get("slice_plane") != "sagittal":
        return _unavailable(
            "L1–S1 腰椎前凸角",
            "需要带方向信息的矢状位影像",
            method="L1 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l1_s1",),
        )
    source_format = str(meta.get("source_format", "")).lower()
    if source_format not in {"nifti", "dicom", "raster"}:
        return _unavailable(
            "L1–S1 腰椎前凸角",
            "输入缺少可用的几何坐标",
            method="L1 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l1_s1",),
        )
    slice_index = _representative_slice(label_maps, l1_label, s1_label)
    if slice_index is None:
        return _unavailable(
            "L1–S1 腰椎前凸角",
            "同一矢状切片未同时获得可靠的 L1、S1 mask",
            method="L1 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l1_s1",),
        )
    label_map = np.asarray(label_maps[slice_index])
    l1_line = _fit_superior_endplate(label_map, l1_label, slice_index, meta)
    s1_line = _fit_superior_endplate(label_map, s1_label, slice_index, meta)
    if l1_line is None or s1_line is None:
        return _unavailable(
            "L1–S1 腰椎前凸角",
            "上终板轮廓不足或拟合质量不合格",
            method="L1 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l1_s1",),
        )
    dot = float(np.clip(abs(np.dot(l1_line[0], s1_line[0])), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))
    raster_note = "，JPG/PNG 方形像素假设" if source_format == "raster" else ""
    return _metric(
        "L1–S1 腰椎前凸角",
        f"{angle:.1f}°（需医师复核{raster_note}）",
        status="measured",
        warn=True,
        method=f"L1 与 S1 上终板 Cobb 角；不自动判定正常/异常{raster_note}",
        reference_ids=("lordosis_l1_s1",),
        numeric_value=angle,
        unit="°",
    )


def compute_l4_s1_lordosis(
    label_maps: Sequence[np.ndarray],
    meta: Optional[Dict[str, Any]],
    *,
    l4_label: int = 4,
    s1_label: int = 6,
) -> Dict[str, Any]:
    """Measure lower lumbar lordosis between the L4 and S1 superior endplates."""
    meta = dict(meta or {})
    if not bool(meta.get("label_semantics_confirmed", False)):
        return _unavailable(
            "L4–S1 下腰椎前凸角",
            "需先确认 label 4=L4、label 6=S1 的正式标签字典",
            method="L4 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l4_s1",),
        )
    if meta.get("slice_plane") != "sagittal":
        return _unavailable(
            "L4–S1 下腰椎前凸角",
            "需要带方向信息的矢状位影像",
            method="L4 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l4_s1",),
        )
    source_format = str(meta.get("source_format", "")).lower()
    if source_format not in {"nifti", "dicom", "raster"}:
        return _unavailable(
            "L4–S1 下腰椎前凸角",
            "输入缺少可用的几何坐标",
            method="L4 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l4_s1",),
        )
    slice_index = _representative_slice(label_maps, l4_label, s1_label)
    if slice_index is None:
        return _unavailable(
            "L4–S1 下腰椎前凸角",
            "同一矢状切片未同时获得可靠的 L4、S1 mask",
            method="L4 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l4_s1",),
        )
    label_map = np.asarray(label_maps[slice_index])
    l4_line = _fit_superior_endplate(label_map, l4_label, slice_index, meta)
    s1_line = _fit_superior_endplate(label_map, s1_label, slice_index, meta)
    if l4_line is None or s1_line is None:
        return _unavailable(
            "L4–S1 下腰椎前凸角",
            "上终板轮廓不足或拟合质量不合格",
            method="L4 与 S1 上终板 Cobb 角",
            reference_ids=("lordosis_l4_s1",),
        )
    dot = float(np.clip(abs(np.dot(l4_line[0], s1_line[0])), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))
    raster_note = "，JPG/PNG 方形像素假设" if source_format == "raster" else ""
    return _metric(
        "L4–S1 下腰椎前凸角",
        f"{angle:.1f}°（体位相关，需复核{raster_note}）",
        status="measured",
        warn=True,
        method=f"L4 与 S1 上终板 Cobb 角；不自动判定正常/异常{raster_note}",
        reference_ids=("lordosis_l4_s1",),
        numeric_value=angle,
        unit="°",
    )


def compute_sacral_slope(
    label_maps: Sequence[np.ndarray],
    meta: Optional[Dict[str, Any]],
    *,
    s1_label: int = 6,
) -> Dict[str, Any]:
    """Measure the S1 superior endplate angle relative to patient AP horizontal."""
    meta = dict(meta or {})
    if not bool(meta.get("label_semantics_confirmed", False)):
        return _unavailable(
            "S1 骶骨倾斜角",
            "需先确认 label 6=S1 的正式标签字典",
            method="S1 上终板与水平线夹角",
            reference_ids=("sacral_slope",),
        )
    if meta.get("slice_plane") != "sagittal":
        return _unavailable(
            "S1 骶骨倾斜角",
            "需要带方向信息的矢状位影像",
            method="S1 上终板与水平线夹角",
            reference_ids=("sacral_slope",),
        )
    source_format = str(meta.get("source_format", "")).lower()
    if source_format == "raster" and not bool(
        meta.get("raster_patient_horizontal_confirmed", False)
    ):
        return _unavailable(
            "S1 骶骨倾斜角",
            "JPG/PNG 无法确认患者水平轴；需原始几何或人工确认",
            method="S1 上终板与水平线夹角",
            reference_ids=("sacral_slope",),
        )
    if source_format not in {"nifti", "dicom", "raster"}:
        return _unavailable(
            "S1 骶骨倾斜角",
            "PNG 缺少患者物理方向",
            method="S1 上终板与水平线夹角",
            reference_ids=("sacral_slope",),
        )
    slice_index = _representative_single_label_slice(label_maps, s1_label)
    if slice_index is None:
        return _unavailable(
            "S1 骶骨倾斜角",
            "未获得可靠的 S1 mask",
            method="S1 上终板与水平线夹角",
            reference_ids=("sacral_slope",),
        )
    s1_line = _fit_superior_endplate(
        np.asarray(label_maps[slice_index]), s1_label, slice_index, meta
    )
    if s1_line is None:
        return _unavailable(
            "S1 骶骨倾斜角",
            "S1 上终板轮廓不足或拟合质量不合格",
            method="S1 上终板与水平线夹角",
            reference_ids=("sacral_slope",),
        )
    dot = float(np.clip(abs(np.dot(s1_line[0], np.asarray([1.0, 0.0]))), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))
    return _metric(
        "S1 骶骨倾斜角",
        f"{angle:.1f}°（体位相关，需复核）",
        status="measured",
        warn=True,
        method="S1 上终板与患者前后轴水平线夹角；不套用站立位正常范围",
        reference_ids=("sacral_slope",),
        numeric_value=angle,
        unit="°",
    )


def _spacing_values(meta: Dict[str, Any]) -> Tuple[List[float], str]:
    values = meta.get("spacing_values_mm")
    if values is not None:
        try:
            parsed = [float(value) for value in values]
            if len(parsed) >= 3 and all(value > 0 for value in parsed[:3]):
                return parsed[:3], "volume"
            if len(parsed) >= 2 and all(value > 0 for value in parsed[:2]):
                return parsed[:2], "area"
        except Exception:
            pass
    return [], "pixel"


def compute_canal_mask_measurement(
    label_maps: Sequence[np.ndarray],
    meta: Optional[Dict[str, Any]],
    *,
    canal_labels: Sequence[int] = (11, 12),
) -> Dict[str, Any]:
    """Return a descriptive canal-mask size, never a stenosis diagnosis."""
    meta = dict(meta or {})
    pixel_count = sum(
        int(np.count_nonzero(np.isin(label_map, canal_labels))) for label_map in label_maps
    )
    if pixel_count == 0:
        return _unavailable(
            "椎管 mask 量化",
            "未检测到椎管 mask",
            method="分割体素数乘物理体素尺寸",
            reference_ids=(),
        )
    spacing, mode = _spacing_values(meta)
    if mode == "volume":
        volume_cm3 = pixel_count * spacing[0] * spacing[1] * spacing[2] / 1000.0
        return _metric(
            "椎管 mask 量化",
            f"{volume_cm3:.2f} cm³（描述性）",
            status="measured",
            warn=False,
            method="椎管 mask 体素数 × 物理体素体积；不得解释为狭窄分级",
            numeric_value=volume_cm3,
            unit="cm³",
        )
    if mode == "area" and len(label_maps) == 1:
        area_mm2 = pixel_count * spacing[0] * spacing[1]
        return _metric(
            "椎管 mask 量化",
            f"{area_mm2:.1f} mm²（描述性）",
            status="measured",
            warn=False,
            method="椎管 mask 像素数 × 像素面积；不得解释为狭窄分级",
            numeric_value=area_mm2,
            unit="mm²",
        )
    return _unavailable(
        "椎管 mask 量化",
        "缺少完整物理间距，无法换算面积或体积",
        method="分割体素数乘物理体素尺寸",
        reference_ids=(),
    )


def compute_clinical_metrics(
    label_maps: Sequence[np.ndarray],
    meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return the fixed set of safe UI rows in display order."""
    rows = [
        compute_l1_s1_lordosis(label_maps, meta),
        compute_l4_s1_lordosis(label_maps, meta),
        compute_sacral_slope(label_maps, meta),
        compute_canal_mask_measurement(label_maps, meta),
        _unavailable(
            "椎间盘高度/退变",
            "当前模型无椎间盘 mask 和经验证的终板标志点",
            method="椎间盘轮廓或标准化终板标志点测量",
            reference_ids=("disc_height",),
        ),
        _unavailable(
            "椎管狭窄分级",
            "需要轴位 T2、节段定位及硬膜囊/神经根/脑脊液形态",
            method="Schizas 轴位 T2 形态学分级",
            reference_ids=("stenosis_schizas", "stenosis_wfns"),
        ),
        _unavailable(
            "骨密度/骨质疏松",
            "需要保留 HU 的 CT 椎体松质骨 ROI 或 DXA",
            method="CT L1 松质骨 HU 或 DXA",
            reference_ids=("ct_bmd",),
        ),
        _unavailable(
            "椎体压缩性骨折",
            "需要前/中/后缘高度标志点并排除非骨折性变形",
            method="Genant 半定量椎体骨折评估",
            reference_ids=("genant",),
        ),
    ]
    return rows
