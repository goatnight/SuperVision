"""
SpineVision AI — 脊柱智能分割系统 (PyQt5)
背景 + 12 个前景 mask 的分割前端界面

功能完善版：
  - 真实数据导入 (PNG/JPG/BMP + NIfTI .nii/.nii.gz + DICOM .dcm 多切片)
  - 切片滑动条同步显示各切片，滚轮翻片
  - 导出分割结果 (PNG序列 / 叠加PNG / NIfTI标签体积)
  - 模型选择支持从文件系统导入 .pth/.pt/.onnx 权重
  - 序列列表动态管理（添加/右键删除/切换）
  - 透明度滑条实时混合原图+分割结果
  - 底部信息栏实时更新
  - 快捷键: Ctrl+O 导入, Ctrl+S 导出, Ctrl+R 开始分割
"""

import sys
import os
import time
import importlib.util
import traceback
import numpy as np
from pathlib import Path

from spinevision_clinical_metrics import (
    compute_clinical_metrics,
    infer_dicom_slice_plane,
    infer_nifti_slice_plane,
)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QFileDialog, QFrame,
    QScrollArea, QProgressBar, QSizePolicy, QListWidget, QListWidgetItem,
    QMessageBox, QDialog, QDialogButtonBox, QLineEdit, QTextEdit,
    QMenu, QAction, QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
    QFont, QPixmap, QImage, QIcon, QPalette, QCursor, QKeySequence
)

# ── Optional heavy deps (graceful degradation) ─────────────────────────────
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ─── Color Palette ──────────────────────────────────────────────────────────
C = {
    "bg_deep":    "#020c18",
    "bg_panel":   "#040f1f",
    "bg_card":    "#071526",
    "bg_card2":   "#0a1e35",
    "cyan":       "#00d4ff",
    "cyan_dim":   "#0099bb",
    "teal":       "#00ffcc",
    "blue":       "#1a6fff",
    "orange":     "#ff6b2b",
    "green":      "#00ff88",
    "text_bright":"#e8f4ff",
    "text_mid":   "#7ba8cc",
    "text_dim":   "#3a6080",
    "border":     "#0b2d45",
    "border_br":  "#00d4ff55",
}

SEG_CLASSES = [
    ("L1 椎体",      "#ff6b6b", "24.3 cm³"),
    ("L2 椎体",      "#ff9f43", "26.1 cm³"),
    ("L3 椎体",      "#ffd32a", "27.8 cm³"),
    ("L4 椎体",      "#a3cb38", "29.2 cm³"),
    ("L5 椎体",      "#0be881", "28.7 cm³"),
    ("S1",           "#00d2d3", "18.4 cm³"),
    ("S2",           "#48dbfb", "14.2 cm³"),
    ("S3",           "#54a0ff", "11.8 cm³"),
    ("S4",           "#5f27cd", " 9.4 cm³"),
    ("S5",           "#9b59b6", " 7.1 cm³"),
    ("椎管 (L1-L3)", "#ff4757", "12.6 cm³"),
    ("椎管 (L4-L5)", "#ff6b81", " 8.9 cm³"),
]

SEG_COLORS_RGB = [
    (255,107,107),(255,159,67),(255,211,42),(163,203,56),(11,232,129),
    (0,210,211),(72,219,251),(84,160,255),(95,39,205),(155,89,182),
    (255,71,87),(255,107,129),
]

BACKGROUND_LABEL = 0
FOREGROUND_CLASS_COUNT = 12
MODEL_TOTAL_CLASSES = FOREGROUND_CLASS_COUNT + 1
TRAINED_INPUT_DOMAIN = "训练输入格式"
COMPATIBLE_INPUT_DOMAIN = "兼容输入（待验证）"
DEFAULT_WEIGHT_PATH = os.environ.get("SPINE_DEFAULT_WEIGHT", "")
LABEL_SEMANTICS_CONFIRMED = os.environ.get(
    "SPINE_LABEL_SEMANTICS_CONFIRMED", "1"
).strip().lower() in ("1", "true", "yes", "on")

if len(SEG_CLASSES) != FOREGROUND_CLASS_COUNT or len(SEG_COLORS_RGB) != FOREGROUND_CLASS_COUNT:
    raise RuntimeError("分割类别配置错误：模型应为背景 + 12 个前景 mask。")


def resolve_default_model_path():
    candidates = []
    if DEFAULT_WEIGHT_PATH:
        candidates.append(Path(DEFAULT_WEIGHT_PATH))
    here = Path(__file__).resolve().parent
    candidates.extend([
        here / "runslite_256" / "best.pt",
        here / "weights" / "best.pt",
        Path(r"I:\best.pt"),
    ])
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p)
    return ""


def short_path_label(path, fallback="内置权重"):
    if not path:
        return fallback
    name = Path(path).name
    if len(name) > 32:
        name = name[:14] + "…" + name[-14:]
    return f"📌 {name}"


def label_map_from_color_mask(mask):
    labels = np.zeros(mask.shape[:2], dtype=np.uint8)
    arr = mask.astype(np.int16)
    for idx, color in enumerate(SEG_COLORS_RGB, start=1):
        diff = np.abs(arr - np.array(color, dtype=np.int16)).sum(axis=2)
        labels[diff < 8] = idx
    return labels


def _class_pixel_counts_from_labels(label_maps):
    counts = np.zeros(len(SEG_CLASSES) + 1, dtype=np.float64)
    for lab in label_maps or []:
        bc = np.bincount(lab.astype(np.int64).ravel(), minlength=len(SEG_CLASSES) + 1)
        counts[:len(bc)] += bc[:len(counts)]
    return counts


def _parse_spacing(meta):
    text = str((meta or {}).get("spacing", "")).replace("×", "x").replace("*", "x")
    vals = []
    for part in text.replace("mm", "").split("x"):
        try:
            vals.append(float(part.strip()))
        except Exception:
            pass
    if len(vals) >= 3:
        return vals[0], vals[1], vals[2], "volume"
    if len(vals) >= 2:
        return vals[0], vals[1], 1.0, "area"
    return 1.0, 1.0, 1.0, "pixel"


def _measure_text(pixel_count, spacing):
    sx, sy, sz, mode = spacing
    if mode == "volume":
        return f"{pixel_count * sx * sy * sz / 1000.0:.2f} cm³"
    if mode == "area":
        return f"{pixel_count * sx * sy:.1f} mm²"
    return f"{int(pixel_count)} px"


def analyze_segmentation(orig_arrays, label_maps, meta=None, clinical_label_maps=None):
    counts = _class_pixel_counts_from_labels(label_maps)
    spacing = _parse_spacing(meta)
    total_pixels = float(sum(lab.size for lab in label_maps)) if label_maps else 0.0
    fg = float(counts[1:].sum())
    detected_classes = int(np.count_nonzero(counts[1:] > 0))
    coverage = fg / max(total_pixels, 1.0)
    clinical_maps = clinical_label_maps if clinical_label_maps is not None else label_maps
    rows = compute_clinical_metrics(clinical_maps or [], meta)
    recommendations = [
        "本页仅显示结构化测量及其可用性，不自动生成临床诊断或治疗方案",
        "前凸角与骶骨倾斜角仅在矢状位且终板拟合合格时输出，并受扫描体位影响",
        "椎管狭窄需轴位 T2 的节段性硬膜囊、神经根和脑脊液形态，当前 mask 不足以分级",
        "骨密度需保留 HU 的 CT 松质骨 ROI 或 DXA；窗口化 DICOM、MRI 和 PNG 不可替代",
    ]
    class_rows = []
    for idx, (name, _, _) in enumerate(SEG_CLASSES, start=1):
        count = float(counts[idx]) if idx < len(counts) else 0.0
        class_rows.append({
            "name": name,
            "pixels": int(count),
            "measure": _measure_text(count, spacing),
            "ratio": count / max(fg, 1.0),
        })

    lordosis = rows[0]
    canal_measure = rows[3]
    summary_metrics = [
        {
            "value": str(detected_classes),
            "unit": "/12",
            "label": "检测到的前景类别",
            "trend": "模型输出，非质量评分",
            "ok": detected_classes > 0,
        },
        {
            "value": f"{coverage*100:.2f}",
            "unit": "%",
            "label": "前景像素占比",
            "trend": "描述性统计，非质量评分",
            "ok": fg > 0,
        },
        {
            "value": "--" if lordosis.get("numeric_value") is None else f"{lordosis['numeric_value']:.1f}",
            "unit": "°",
            "label": "L1–S1 腰椎前凸角",
            "trend": "需医师复核" if lordosis.get("status") == "measured" else "输入条件不足",
            "ok": lordosis.get("status") == "measured",
        },
        {
            "value": "--" if canal_measure.get("numeric_value") is None else f"{canal_measure['numeric_value']:.2f}",
            "unit": canal_measure.get("unit", ""),
            "label": "椎管 mask 量化",
            "trend": "描述性，非狭窄分级" if canal_measure.get("status") == "measured" else "输入条件不足",
            "ok": canal_measure.get("status") == "measured",
        },
    ]

    return {
        "rows": rows,
        "recommendations": recommendations,
        "counts": counts.tolist(),
        "class_rows": class_rows,
        "summary_metrics": summary_metrics,
        "coverage": coverage,
        "risk_score": None,
        "spacing_mode": spacing[3],
        "clinical_references": sorted(
            {
                reference_id
                for row in rows
                for reference_id in row.get("reference_ids", [])
            }
        ),
    }


def _env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def apply_medical_orientation(arr):
    """Unify displayed/imported medical-image orientation.

    The original UI read DICOM directly and used np.rot90 for NIfTI, which can
    make sagittal spine data appear mirrored in both axes on some datasets.  The
    default correction flips left-right and up-down for DICOM/NIfTI only.  If a
    dataset has a different convention, override with environment variables:
      SPINE_FLIP_LR=0/1, SPINE_FLIP_UD=0/1, SPINE_ROT90=0/1/2/3
    """
    out = arr
    rot_k = int(os.environ.get("SPINE_ROT90", "0")) % 4
    if rot_k:
        out = np.rot90(out, k=rot_k)
    # if _env_flag("SPINE_FLIP_LR", True):
    #     out = np.fliplr(out)
    if _env_flag("SPINE_FLIP_UD", True):
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def inverse_medical_orientation(arr):
    out = arr
    if _env_flag("SPINE_FLIP_UD", True):
        out = np.flipud(out)
    rot_k = int(os.environ.get("SPINE_ROT90", "0")) % 4
    if rot_k:
        out = np.rot90(out, k=-rot_k)
    return np.ascontiguousarray(out)


def nifti_slice_to_display(arr):
    return apply_medical_orientation(np.rot90(arr))


def display_to_nifti_slice(arr):
    return np.ascontiguousarray(np.rot90(inverse_medical_orientation(arr), k=-1))


def rgb_to_gray_float(arr):
    if arr.ndim == 2:
        return arr.astype(np.float32)
    return (
        0.299 * arr[..., 0].astype(np.float32)
        + 0.587 * arr[..., 1].astype(np.float32)
        + 0.114 * arr[..., 2].astype(np.float32)
    )


def add_low_field_white_noise(arr):
    if not _env_flag("SPINE_LOW_FIELD_NOISE", True):
        return np.ascontiguousarray(arr)
    std = float(os.environ.get("SPINE_LOW_FIELD_NOISE_STD", "20.0"))
    if std <= 0:
        return np.ascontiguousarray(arr)
    src = arr.astype(np.float32, copy=False)
    rng = np.random.default_rng()
    noise = rng.normal(0.0, std, src.shape[:2]).astype(np.float32)
    if src.ndim == 3:
        noise = noise[..., None]
    out = np.clip(src + noise, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def labels_to_color_mask(label_map):
    mask = np.zeros(label_map.shape[:2] + (3,), dtype=np.uint8)
    for label_id, color in enumerate(SEG_COLORS_RGB, start=1):
        mask[label_map == label_id] = color
    return mask


STYLESHEET = f"""
QWidget {{
    background-color: {C['bg_deep']};
    color: {C['text_bright']};
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
}}
QScrollBar:vertical {{
    background: {C['bg_panel']}; width: 6px; border: none;
}}
QScrollBar::handle:vertical {{
    background: {C['cyan_dim']}; border-radius: 3px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {C['bg_panel']}; height: 6px; border: none;
}}
QScrollBar::handle:horizontal {{
    background: {C['cyan_dim']}; border-radius: 3px;
}}
QComboBox {{
    background: {C['bg_card']}; border: 1px solid {C['border']};
    color: {C['text_bright']}; padding: 5px 10px; font-size: 12px;
    selection-background-color: {C['bg_card2']};
}}
QComboBox:hover {{ border-color: {C['cyan_dim']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {C['bg_card']}; border: 1px solid {C['border']};
    color: {C['text_bright']}; selection-background-color: {C['bg_card2']};
}}
QSlider::groove:horizontal {{
    height: 3px; background: {C['border']}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C['cyan']}; border: none;
    width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C['cyan']}, stop:1 {C['teal']});
    border-radius: 2px;
}}
QProgressBar {{
    background: {C['bg_card']}; border: 1px solid {C['border']};
    height: 4px; border-radius: 2px; text-align: center; font-size: 9px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C['cyan']}, stop:1 {C['teal']});
    border-radius: 2px;
}}
QListWidget {{
    background: transparent; border: none; outline: none; font-size: 12px;
}}
QListWidget::item {{
    padding: 6px 8px; border-left: 2px solid transparent;
}}
QListWidget::item:hover {{
    background: rgba(0,140,200,50); border-left-color: {C['cyan_dim']};
}}
QListWidget::item:selected {{
    background: rgba(0,212,255,30); border-left-color: {C['cyan']};
    color: {C['text_bright']};
}}
QLineEdit {{
    background: {C['bg_card']}; border: 1px solid {C['border']};
    color: {C['text_bright']}; padding: 5px 8px; font-size: 12px;
}}
QLineEdit:focus {{ border-color: {C['cyan_dim']}; }}
QTextEdit {{
    background: {C['bg_card']}; border: 1px solid {C['border']};
    color: {C['text_mid']}; font-size: 11px;
}}
QMenu {{
    background: {C['bg_card']}; border: 1px solid {C['border']};
    color: {C['text_bright']}; padding: 4px;
}}
QMenu::item {{ padding: 6px 20px; }}
QMenu::item:selected {{ background: rgba(0,212,255,40); color: {C['cyan']}; }}
"""


# ─── Data helpers ────────────────────────────────────────────────────────────

def load_image_file(path: str):
    """
    Returns (slices: list[np.ndarray HxWx3 uint8], meta: dict) or (None, meta).
    Supports PNG/JPG/BMP (single), NIfTI (multi-slice), DICOM (single).
    """
    ext = Path(path).suffix.lower()
    meta = {
        "path": path, "filename": Path(path).name,
        "modality": "—", "shape": "—", "spacing": "—",
        "model_input_domain": COMPATIBLE_INPUT_DOMAIN,
        "source_format": "unknown", "slice_plane": "unknown",
        "label_semantics_confirmed": LABEL_SEMANTICS_CONFIRMED,
    }

    # ── Plain image ──────────────────────────────────────────────────
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        qpm = QPixmap(path)
        if qpm.isNull():
            return None, meta
        img = qpm.toImage().convertToFormat(QImage.Format_RGB888)
        w, h = img.width(), img.height()
        ptr = img.bits(); ptr.setsize(h * w * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3)).copy()
        infer_gray = rgb_to_gray_float(arr)
        if _env_flag("SPINE_PNG_MATCH_NIFTI_VIEW", True):
            infer_gray = display_to_nifti_slice(infer_gray)
            meta["infer_display_transform"] = "nifti"
        meta["infer_arrays"] = [np.ascontiguousarray(infer_gray, dtype=np.float32)]
        arr = add_low_field_white_noise(arr)
        meta.update({"modality": ext.upper().strip("."),
                     "shape": f"{w}×{h}×1", "spacing": "—",
                     "model_input_domain": COMPATIBLE_INPUT_DOMAIN,
                     "source_format": "raster",
                     "slice_plane": "sagittal" if _env_flag("SPINE_RASTER_ASSUME_SAGITTAL", True) else "unknown",
                     "spacing_values_mm": [],
                     "measurement_geometry": "pixel_assumed_square",
                     "raster_patient_horizontal_confirmed": _env_flag(
                         "SPINE_RASTER_PATIENT_HORIZONTAL_CONFIRMED", False
                     )})
        return [arr], meta

    # ── NIfTI ────────────────────────────────────────────────────────
    if ext in (".nii", ".gz"):
        if not HAS_NIBABEL:
            QMessageBox.warning(None, "缺少依赖",
                "读取 NIfTI 需要安装 nibabel:\n  pip install nibabel")
            return None, meta
        try:
            nii_img = nib.load(path)
            raw_data = nii_img.get_fdata(dtype=np.float32)
            if raw_data.ndim == 2:
                raw_data = raw_data[:, :, np.newaxis]
            elif raw_data.ndim == 4:
                raw_data = raw_data[:, :, :, 0]
            display_data = raw_data
            lo = float(np.percentile(display_data, 1))
            hi = float(np.percentile(display_data, 99))
            display_data = np.clip(display_data, lo, hi)
            display_data = ((display_data - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
            slices = []
            infer_arrays = []
            for z in range(raw_data.shape[2]):
                infer_arrays.append(np.ascontiguousarray(raw_data[:, :, z], dtype=np.float32))
                sl = nifti_slice_to_display(display_data[:, :, z])
                rgb = np.stack([sl, sl, sl], axis=-1)
                rgb = add_low_field_white_noise(rgb)
                slices.append(rgb)
            hdr = nii_img.header
            zooms = hdr.get_zooms()[:3]
            sp = "×".join(f"{float(z):.2f}" for z in zooms) + " mm"
            x, y, nz = raw_data.shape
            affine = np.asarray(nii_img.affine, dtype=np.float64)
            meta.update({"modality": "NIfTI",
                         "shape": f"{x}×{y}×{nz}", "spacing": sp,
                         "model_input_domain": TRAINED_INPUT_DOMAIN,
                         "source_format": "nifti",
                         "slice_plane": infer_nifti_slice_plane(affine),
                         "spacing_values_mm": [float(z) for z in zooms],
                         "nifti_affine": affine.tolist(),
                         "pixel_values_in_hu": False})
            meta["infer_arrays"] = infer_arrays
            meta["infer_display_transform"] = "nifti"
            return slices, meta
        except Exception as e:
            QMessageBox.warning(None, "NIfTI 读取失败", str(e))
            return None, meta

    # ── DICOM ────────────────────────────────────────────────────────
    if ext == ".dcm":
        if not HAS_PYDICOM:
            QMessageBox.warning(None, "缺少依赖",
                "读取 DICOM 需要安装 pydicom:\n  pip install pydicom")
            return None, meta
        try:
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
            wc = getattr(ds, "WindowCenter", None)
            ww = getattr(ds, "WindowWidth", None)
            if wc is not None and ww is not None:
                wc = float(wc) if not hasattr(wc, "__iter__") else float(wc[0])
                ww = float(ww) if not hasattr(ww, "__iter__") else float(ww[0])
                lo_, hi_ = wc - ww / 2, wc + ww / 2
            else:
                lo_, hi_ = float(arr.min()), float(arr.max())
            arr = np.clip(arr, lo_, hi_)
            arr = ((arr - lo_) / max(hi_ - lo_, 1e-6) * 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            arr = apply_medical_orientation(arr)
            spacing = "—"
            spacing_values = []
            if hasattr(ds, "PixelSpacing"):
                ps = ds.PixelSpacing
                spacing = f"{float(ps[0]):.2f}×{float(ps[1]):.2f} mm"
                spacing_values = [float(ps[0]), float(ps[1])]
            modality = getattr(ds, "Modality", "DICOM")
            orientation = [float(v) for v in getattr(ds, "ImageOrientationPatient", [])]
            position = [float(v) for v in getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])]
            h2, w2 = arr.shape[:2]
            meta.update({"modality": modality,
                         "shape": f"{w2}×{h2}×1", "spacing": spacing,
                         "model_input_domain": COMPATIBLE_INPUT_DOMAIN,
                         "source_format": "dicom",
                         "slice_plane": infer_dicom_slice_plane(orientation),
                         "spacing_values_mm": spacing_values,
                         "dicom_pixel_spacing": spacing_values,
                         "dicom_image_orientation": orientation,
                         "dicom_image_positions": [position],
                         "pixel_values_in_hu": False})
            return [arr], meta
        except Exception as e:
            QMessageBox.warning(None, "DICOM 读取失败", str(e))
            return None, meta

    QMessageBox.warning(None, "不支持的格式",
                        f"不支持的文件格式: {ext}\n"
                        f"支持: .png .jpg .bmp .tif .nii .nii.gz .dcm")
    return None, meta


def load_dicom_folder(folder: str):
    """Load an entire DICOM folder as a multi-slice volume."""
    if not HAS_PYDICOM:
        QMessageBox.warning(None, "缺少依赖",
            "读取 DICOM 需要安装 pydicom:\n  pip install pydicom")
        return None, {}
    dcm_files = sorted(Path(folder).glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(Path(folder).glob("**/*.dcm"))
    if not dcm_files:
        QMessageBox.warning(None, "未找到 DICOM",
                            f"文件夹中未找到 .dcm 文件:\n{folder}")
        return None, {}
    slices_raw = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            slices_raw.append((getattr(ds, "InstanceNumber", 0), ds))
        except Exception:
            pass
    slices_raw.sort(key=lambda x: x[0])
    slices = []
    for _, ds in slices_raw:
        arr = ds.pixel_array.astype(np.float32)
        lo_, hi_ = float(arr.min()), float(arr.max())
        arr = np.clip(arr, lo_, hi_)
        arr = ((arr - lo_) / max(hi_ - lo_, 1e-6) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = apply_medical_orientation(arr)
        slices.append(arr)
    if not slices:
        return None, {}
    ds0 = slices_raw[0][1]
    spacing = "—"
    spacing_values = []
    if hasattr(ds0, "PixelSpacing"):
        ps = ds0.PixelSpacing
        slice_spacing = None
        positions_for_spacing = []
        orientation_for_spacing = [float(v) for v in getattr(ds0, "ImageOrientationPatient", [])]
        if len(orientation_for_spacing) == 6:
            normal = np.cross(
                np.asarray(orientation_for_spacing[:3], dtype=np.float64),
                np.asarray(orientation_for_spacing[3:], dtype=np.float64),
            )
            for _, ds in slices_raw:
                position = getattr(ds, "ImagePositionPatient", None)
                if position is not None:
                    positions_for_spacing.append(
                        float(np.dot(np.asarray(position, dtype=np.float64), normal))
                    )
        if len(positions_for_spacing) >= 2:
            diffs = np.abs(np.diff(sorted(positions_for_spacing)))
            positive_diffs = diffs[diffs > 1e-6]
            if positive_diffs.size:
                slice_spacing = float(np.median(positive_diffs))
        if slice_spacing is None:
            try:
                slice_spacing = float(getattr(ds0, "SpacingBetweenSlices"))
            except Exception:
                try:
                    slice_spacing = float(getattr(ds0, "SliceThickness"))
                except Exception:
                    slice_spacing = None
        spacing_values = [float(ps[0]), float(ps[1])]
        if slice_spacing is not None and slice_spacing > 0:
            spacing_values.append(slice_spacing)
            spacing = f"{float(ps[0]):.2f}×{float(ps[1]):.2f}×{slice_spacing:.2f} mm"
        else:
            spacing = f"{float(ps[0]):.2f}×{float(ps[1]):.2f} mm"
    h2, w2 = slices[0].shape[:2]
    orientation = [float(v) for v in getattr(ds0, "ImageOrientationPatient", [])]
    positions = [
        [float(v) for v in getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])]
        for _, ds in slices_raw
    ]
    meta = {
        "path": folder, "filename": Path(folder).name,
        "modality": getattr(ds0, "Modality", "DICOM"),
        "shape": f"{w2}×{h2}×{len(slices)}", "spacing": spacing,
        "model_input_domain": COMPATIBLE_INPUT_DOMAIN,
        "source_format": "dicom",
        "slice_plane": infer_dicom_slice_plane(orientation),
        "spacing_values_mm": spacing_values,
        "dicom_pixel_spacing": spacing_values[:2],
        "dicom_image_orientation": orientation,
        "dicom_image_positions": positions,
        "pixel_values_in_hu": False,
    }
    return slices, meta


def numpy_to_pixmap(arr: np.ndarray) -> QPixmap:
    h, w = arr.shape[:2]
    c_arr = np.ascontiguousarray(arr)
    img = QImage(c_arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def blend_overlay(orig: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    result = orig.astype(np.float32) * (1.0 - alpha) + mask.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def make_fake_seg_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    n = len(SEG_COLORS_RGB)
    for i, (r, g, b) in enumerate(SEG_COLORS_RGB):
        y0 = int(h * i / n)
        y1 = int(h * (i + 1) / n)
        x0, x1 = int(w * 0.18), int(w * 0.82)
        mask[y0:y1, x0:x1] = (r, g, b)
    return mask


# ─── Custom Widgets ─────────────────────────────────────────────────────────

class CyberButton(QPushButton):
    def __init__(self, text, accent=False, parent=None):
        super().__init__(text, parent)
        self.accent = accent
        self.setFixedHeight(34)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._update_style(False)

    def _update_style(self, hovered):
        if self.accent:
            bg = "rgba(0,100,180,180)" if not hovered else "rgba(0,140,210,210)"
            border = C['cyan'] if hovered else C['cyan_dim']
            color = C['text_bright']; fsize = "14px"
        else:
            bg = "rgba(0,30,60,140)" if not hovered else "rgba(0,60,100,180)"
            border = C['cyan_dim'] if hovered else C['border']
            color = C['cyan'] if hovered else C['text_mid']; fsize = "12px"
        self.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; border:1px solid {border};
                color:{color}; padding:6px 18px;
                font-family:"Rajdhani","Microsoft YaHei",sans-serif;
                font-size:{fsize}; font-weight:bold; letter-spacing:2px;
            }}
            QPushButton:pressed {{ background:rgba(0,212,255,40); }}
            QPushButton:disabled {{ color:{C['text_dim']}; border-color:{C['border']}; }}
        """)

    def enterEvent(self, e): self._update_style(True)
    def leaveEvent(self, e): self._update_style(False)


class ImageViewport(QWidget):
    """Viewport with scan-line animation, pan, zoom, and multi-slice support."""
    slice_changed = pyqtSignal(int)

    def __init__(self, placeholder_icon="🦴", placeholder_text="暂无影像数据", parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self._slices = []
        self._cur = 0
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self._drag_start = None
        self.placeholder_icon = placeholder_icon
        self.placeholder_text = placeholder_text
        self.mouse_pos = QPoint(-1, -1)
        self._scan_y = 0
        t = QTimer(self); t.timeout.connect(self._tick); t.start(16)

    def _tick(self):
        self._scan_y = (self._scan_y + 1) % max(1, self.height())
        self.update()

    def set_slices(self, pixmaps):
        self._slices = pixmaps; self._cur = 0
        self.scale = 1.0; self.offset = QPoint(0, 0); self.update()

    def set_slice_index(self, idx):
        if self._slices:
            self._cur = max(0, min(idx, len(self._slices) - 1))
            self.update()

    def current_pixmap(self):
        return self._slices[self._cur] if self._slices else None

    def n_slices(self): return len(self._slices)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # Background + grid
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#040f1f")); grad.setColorAt(1, QColor("#020c18"))
        p.fillRect(0, 0, w, h, grad)
        p.setPen(QPen(QColor(0, 100, 160, 12)))
        for x in range(0, w, 30): p.drawLine(x, 0, x, h)
        for y in range(0, h, 30): p.drawLine(0, y, w, y)
        rg = QRadialGradient(w/2, h/2, w*0.6)
        rg.setColorAt(0, QColor(0, 60, 120, 25)); rg.setColorAt(1, QColor(0,0,0,0))
        p.fillRect(0, 0, w, h, rg)

        pm = self.current_pixmap()
        if pm:
            scaled = pm.scaled(int(pm.width()*self.scale), int(pm.height()*self.scale),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
            px = (w - scaled.width())  // 2 + self.offset.x()
            py = (h - scaled.height()) // 2 + self.offset.y()
            p.drawPixmap(px, py, scaled)
            if len(self._slices) > 1:
                txt = f"Slice {self._cur+1}/{len(self._slices)}"
                p.setFont(QFont("Courier New", 9))
                p.setPen(QColor(C['cyan_dim']))
                p.fillRect(w-112, 8, 104, 16, QColor(2,12,24,180))
                p.drawText(QRect(w-112, 8, 104, 16), Qt.AlignCenter, txt)
        else:
            p.setFont(QFont("Segoe UI Emoji", 36))
            p.setPen(QColor(C['text_dim']))
            p.drawText(QRect(0, h//2-80, w, 60), Qt.AlignCenter, self.placeholder_icon)
            p.setFont(QFont("Microsoft YaHei", 11))
            p.drawText(QRect(0, h//2-10, w, 30), Qt.AlignCenter, self.placeholder_text)
            p.setFont(QFont("Courier New", 9))
            p.drawText(QRect(0, h//2+22, w, 20), Qt.AlignCenter, "双击放大 · Ctrl+滚轮缩放")

        # Scan line
        sg = QLinearGradient(0, self._scan_y, w, self._scan_y)
        sg.setColorAt(0, QColor(0,212,255,0)); sg.setColorAt(0.5, QColor(0,212,255,80))
        sg.setColorAt(1, QColor(0,212,255,0))
        p.setPen(QPen(QBrush(sg), 1.5)); p.drawLine(0, self._scan_y, w, self._scan_y)

        # Corners
        p.setPen(QPen(QColor(C['cyan']), 2))
        cs, m = 14, 8
        p.drawLine(m, m, m+cs, m); p.drawLine(m, m, m, m+cs)
        p.drawLine(w-m-cs, m, w-m, m); p.drawLine(w-m, m, w-m, m+cs)
        p.drawLine(m, h-m, m+cs, h-m); p.drawLine(m, h-m-cs, m, h-m)
        p.drawLine(w-m-cs, h-m, w-m, h-m); p.drawLine(w-m, h-m-cs, w-m, h-m)

        # Coord info
        if self.mouse_pos.x() >= 0:
            info = f"X:{self.mouse_pos.x()}  Y:{self.mouse_pos.y()}"
            p.setFont(QFont("Courier New", 9)); p.setPen(QColor(C['cyan_dim']))
            p.fillRect(m+2, h-24, 160, 16, QColor(2,12,24,180))
            p.drawText(m+6, h-12, info)

    def mouseMoveEvent(self, e):
        self.mouse_pos = e.pos()
        if self._drag_start and e.buttons() & Qt.LeftButton:
            self.offset += e.pos() - self._drag_start
            self._drag_start = e.pos()
        self.update()

    def mousePressEvent(self, e): self._drag_start = e.pos()
    def mouseReleaseEvent(self, e): self._drag_start = None

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            factor = 1.1 if e.angleDelta().y() > 0 else 0.9
            self.scale = max(0.3, min(6.0, self.scale * factor)); self.update()
        else:
            delta = -1 if e.angleDelta().y() > 0 else 1
            new_idx = self._cur + delta
            if self._slices and 0 <= new_idx < len(self._slices):
                self._cur = new_idx
                self.slice_changed.emit(self._cur); self.update()

    def mouseDoubleClickEvent(self, e):
        self.scale = 1.0; self.offset = QPoint(0, 0); self.update()

    def leaveEvent(self, e):
        self.mouse_pos = QPoint(-1, -1); self.update()


class MetricCard(QFrame):
    def __init__(self, value, unit, label, trend="", trend_up=True, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(82)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setStyleSheet(f"""
            QFrame {{
                background:{C['bg_card']}; border:1px solid {C['border']};
                border-top:2px solid {C['cyan']};
            }}
        """)
        lay = QVBoxLayout(self); lay.setContentsMargins(10,10,10,10); lay.setSpacing(4)
        val_lbl = QLabel()
        self.val_lbl = val_lbl
        val_lbl.setTextFormat(Qt.RichText)
        val_lbl.setWordWrap(False)
        val_lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        lay.addWidget(val_lbl)
        lbl = QLabel(label); lbl.setWordWrap(True)
        self.label_lbl = lbl
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        lbl.setStyleSheet(f"font-size:11px;color:{C['text_mid']};background:transparent;border:none;")
        lay.addWidget(lbl)
        self.trend_lbl = None
        if trend:
            tr = QLabel()
            self.trend_lbl = tr
            tr.setWordWrap(True); tr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            lay.addWidget(tr)
        self.set_values(value, unit, label, trend, trend_up)

    def set_values(self, value, unit, label, trend="", trend_up=True):
        self.val_lbl.setText(
            f"<span style='font-size:24px;font-weight:bold;color:{C['text_bright']}'>{value}</span>"
            f"<span style='font-size:12px;color:{C['text_dim']}'>&nbsp;{unit}</span>")
        self.label_lbl.setText(label)
        if self.trend_lbl is not None:
            color = C['green'] if trend_up else C['orange']
            prefix = "▲ " if trend_up else "▼ "
            self.trend_lbl.setText(prefix + trend if trend else "")
            self.trend_lbl.setStyleSheet(f"font-size:10px;color:{color};font-family:'Courier New';"
                                         f"background:transparent;border:none;")


class SectionHeader(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent); self.title = title; self.setFixedHeight(32)

    def paintEvent(self, event):
        p = QPainter(self); w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C['bg_panel']))
        p.fillRect(14, (h-12)//2, 3, 12, QColor(C['cyan']))
        p.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        p.setPen(QColor(C['cyan']))
        p.drawText(QRect(24, 0, w-30, h), Qt.AlignVCenter, self.title.upper())
        p.setPen(QPen(QColor(C['border']), 1)); p.drawLine(0, h-1, w, h-1)


class TopBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setFixedHeight(52); self._anim_x = 0
        t = QTimer(self); t.timeout.connect(self._tick); t.start(20)

    def _tick(self):
        self._anim_x = (self._anim_x + 2) % max(1, self.width()); self.update()

    def paintEvent(self, event):
        p = QPainter(self); w, h = self.width(), self.height()
        g = QLinearGradient(0,0,0,h)
        g.setColorAt(0, QColor(0,25,55,240)); g.setColorAt(1, QColor(4,15,31,200))
        p.fillRect(0,0,w,h,g)
        sg = QLinearGradient(self._anim_x-200, h-1, self._anim_x+200, h-1)
        sg.setColorAt(0, QColor(0,212,255,0)); sg.setColorAt(0.5, QColor(0,212,255,200))
        sg.setColorAt(1, QColor(0,212,255,0))
        p.setPen(QPen(QBrush(sg),1)); p.drawLine(0, h-1, w, h-1)


class SegWorker(QThread):
    """后台真实推理线程：加载 SwinUMambaD 权重，输出彩色分割预测图。"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, orig_arrays, model_path="", meta=None, infer_arrays=None, parent=None):
        super().__init__(parent)
        self.orig_arrays = orig_arrays
        self.infer_arrays = infer_arrays if infer_arrays is not None else orig_arrays
        self.model_path = model_path
        self.meta = meta or {}
        self.image_size = int(os.environ.get("SPINE_INFER_SIZE", "640"))
        self.num_classes = int(os.environ.get("SPINE_NUM_CLASSES", str(MODEL_TOTAL_CLASSES)))
        self._pred_has_background = None
        self._checkpoint_config = {}

    # ── model loading ────────────────────────────────────────────────
    def _candidate_model_py_files(self):
        """不改 UI：通过环境变量或常见路径寻找模型结构 py 文件。"""
        here = Path(__file__).resolve().parent
        candidates = []
        env_path = os.environ.get("SPINE_MODEL_PY", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend([
            here / "SwinUMambaD_prompt_ok.py",
            here / "SwinUMambaD_noCNN.py",
            here / "models" / "SwinUMambaD_prompt_ok.py",
            here / "models" / "SwinUMambaD_noCNN.py",
        ])
        if self.model_path:
            wdir = Path(self.model_path).resolve().parent
            candidates.extend([
                wdir / "SwinUMambaD_prompt_ok.py",
                wdir / "SwinUMambaD_noCNN.py",
                wdir / "models" / "SwinUMambaD_prompt_ok.py",
                wdir / "models" / "SwinUMambaD_noCNN.py",
            ])
        # 去重并只返回存在文件
        seen, found = set(), []
        for p in candidates:
            p = Path(p)
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                seen.add(key)
                if p.exists() and p.is_file():
                    found.append(p)
        return found

    def _import_model_module(self):
        model_files = self._candidate_model_py_files()
        if not model_files:
            raise RuntimeError(
                "没有找到模型结构文件。\n\n"
                "请把 SwinUMambaD_prompt_ok.py 或 SwinUMambaD_noCNN.py 放到本程序同级目录，\n"
                "或设置环境变量 SPINE_MODEL_PY=你的模型结构py完整路径。"
            )
        model_py = model_files[0]
        spec = importlib.util.spec_from_file_location("spine_model_def", str(model_py))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"模型结构文件无法导入: {model_py}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(model_py.parent.parent))
        sys.path.insert(0, str(model_py.parent))
        spec.loader.exec_module(module)
        return module, model_py

    @staticmethod
    def _extract_state_dict(ckpt, torch_module):
        if isinstance(ckpt, dict):
            for key in ("model", "model_state", "state_dict", "model_state_dict", "net", "network", "ema"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
            if all(torch_module.is_tensor(v) for v in ckpt.values()):
                return ckpt
        raise RuntimeError("权重文件里没有找到 state_dict。支持 {'model':...} / {'state_dict':...} / 纯 state_dict。")

    @staticmethod
    def _clean_state_dict_keys(state_dict):
        cleaned = {}
        prefixes = ("module.", "model.", "net.", "network.")
        for k, v in state_dict.items():
            nk = k
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        changed = True
            cleaned[nk] = v
        return cleaned

    @staticmethod
    def _infer_num_classes(state_dict, fallback):
        # 优先从最后分割头权重推断输出通道数，避免 num_classes 手动写错
        for k, v in state_dict.items():
            if ("seg_layers" in k or "seg_layer" in k or "out_conv" in k) and hasattr(v, "shape") and len(v.shape) >= 1:
                c = int(v.shape[0])
                if 1 <= c <= 64:
                    return c
        return fallback

    @staticmethod
    def _attach_selective_scan(model):
        has_ss2d = any(m.__class__.__name__ in ("SS2D", "DAS_SS2D") for m in model.modules())
        if not has_ss2d:
            return
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except Exception as e:
            try:
                from models.mamba_fallback import selective_scan_fn
                print(f"[Model] mamba_ssm unavailable, use PyTorch fallback selective_scan: {e}")
            except Exception as e2:
                raise RuntimeError(
                    "模型包含 SS2D / Mamba 模块，但当前环境无法导入 mamba_ssm 或 fallback selective_scan。\n"
                    f"mamba_ssm错误: {e}\n fallback错误: {e2}"
                )
        for m in model.modules():
            if m.__class__.__name__ in ("SS2D", "DAS_SS2D"):
                m.selective_scan = selective_scan_fn

    @staticmethod
    def _torch_load_checkpoint(path, torch_module, map_location="cpu"):
        try:
            return torch_module.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch_module.load(path, map_location=map_location)

    @staticmethod
    def _ensure_train_test_package():
        here = Path(__file__).resolve().parent
        train_root = here / "train_test"
        package_root = train_root / "spinevision"
        if not package_root.exists():
            return None
        train_root_str = str(train_root)
        if train_root_str not in sys.path:
            sys.path.insert(0, train_root_str)
        return train_root

    def _load_train_test_checkpoint(self, ckpt, _torch_module, device):
        if not isinstance(ckpt, dict) or "model_state" not in ckpt or "config" not in ckpt:
            return None

        train_root = self._ensure_train_test_package()
        if train_root is None:
            raise RuntimeError("当前工程中没有找到 train_test/spinevision，无法载入训练得到的 checkpoint。")

        try:
            from spinevision.models import build_model
        except Exception as e:
            raise RuntimeError(f"无法导入 train_test/spinevision 模型代码: {e}") from e

        config = ckpt.get("config") or {}
        model_name = str(config.get("model", "lite"))
        num_classes = int(config.get("num_classes", self.num_classes))
        in_channels = int(config.get("in_channels", 1))
        base_channels = int(config.get("base_channels", 16))
        depth = int(config.get("depth", 1))
        official_base_dim = int(config.get("official_base_dim", 24))

        if "SPINE_INFER_SIZE" not in os.environ and config.get("image_size"):
            self.image_size = int(config["image_size"])
        self.num_classes = num_classes
        self._pred_has_background = True
        self._checkpoint_config = dict(config)

        self.progress.emit(
            22,
            f"初始化 train_test/{model_name}: classes={num_classes}, size={self.image_size}...",
        )
        model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            official_base_dim=official_base_dim,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()

        print(f"[Model] train_test package: {train_root}")
        print(f"[Model] config: {config}")
        print(f"[Model] weight: {self.model_path}")
        print(f"[Model] device: {device}")
        return model, device

    def _build_model(self, module, num_classes, ckpt_args=None):
        if not hasattr(module, "SwinUMambaD"):
            raise RuntimeError("模型结构文件中没有找到 SwinUMambaD 类。")

        ckpt_args = ckpt_args or {}
        init_type = ckpt_args.get("init_type", "dino")
        if init_type == "dino":
            drop_path_rate = 0.4
        else:
            drop_path_rate = float(ckpt_args.get("drop_path_rate", 0.1))

        vss_args = dict(
            patch_size=int(ckpt_args.get("patch_size", 4)),
            in_chans=int(ckpt_args.get("in_chans", 1)),
            depths=[2, 2, 27, 2],
            dims=[128, 256, 512, 1024],
            d_state=16,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            patch_norm=True,
            use_checkpoint=False,
        )
        decoder_args = dict(
            num_classes=num_classes,
            deep_supervision=False,
            features_per_stage=[128, 256, 512, 1024],
            drop_path_rate=0.2,
            d_state=16,
        )
        return module.SwinUMambaD(vss_args, decoder_args)

    def _load_model(self, torch, F):
        if not self.model_path:
            raise RuntimeError("请先在左侧模型参数处选择 .pth / .pt 权重文件。")
        if Path(self.model_path).suffix.lower() == ".onnx":
            raise RuntimeError("当前这版 Worker 实现的是 PyTorch .pth/.pt 推理；ONNX 请另接 onnxruntime 分支。")

        self.progress.emit(8, "读取权重文件...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = self._torch_load_checkpoint(self.model_path, torch, map_location="cpu")
        loaded = self._load_train_test_checkpoint(ckpt, torch, device)
        if loaded is not None:
            return loaded

        self.progress.emit(15, "导入模型结构文件...")
        module, model_py = self._import_model_module()

        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict) else {}
        if "SPINE_INFER_SIZE" not in os.environ and ckpt_args.get("image_size"):
            self.image_size = int(ckpt_args["image_size"])
        state_dict = self._clean_state_dict_keys(self._extract_state_dict(ckpt, torch))
        num_classes = self._infer_num_classes(state_dict, self.num_classes)

        self.progress.emit(22, f"初始化 SwinUMambaD: classes={num_classes}...")
        model = self._build_model(module, num_classes=num_classes, ckpt_args=ckpt_args)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Weight] missing keys: {len(missing)}")
        if unexpected:
            print(f"[Weight] unexpected keys: {len(unexpected)}")

        self._attach_selective_scan(model)
        model.to(device).eval()
        print(f"[Model] structure: {model_py}")
        print(f"[Model] weight: {self.model_path}")
        print(f"[Model] device: {device}")
        return model, device

    # ── preprocessing / postprocessing ────────────────────────────────
    @staticmethod
    def _center_pad_or_crop_np(gray, target_size):
        h, w = gray.shape
        th = tw = int(target_size)
        out = np.zeros((th, tw), dtype=np.float32)

        if h >= th:
            src_y0 = (h - th) // 2
            src_y1 = src_y0 + th
            dst_y0 = 0
            dst_y1 = th
        else:
            src_y0 = 0
            src_y1 = h
            dst_y0 = (th - h) // 2
            dst_y1 = dst_y0 + h

        if w >= tw:
            src_x0 = (w - tw) // 2
            src_x1 = src_x0 + tw
            dst_x0 = 0
            dst_x1 = tw
        else:
            src_x0 = 0
            src_x1 = w
            dst_x0 = (tw - w) // 2
            dst_x1 = dst_x0 + w

        out[dst_y0:dst_y1, dst_x0:dst_x1] = gray[src_y0:src_y1, src_x0:src_x1]
        restore = {
            "orig_hw": (h, w),
            "src": (src_y0, src_y1, src_x0, src_x1),
            "dst": (dst_y0, dst_y1, dst_x0, dst_x1),
        }
        return out, restore

    def _preprocess_slice(self, arr, torch, F, device):
        if arr.ndim == 3:
            gray = arr[..., 0].astype(np.float32)
        else:
            gray = arr.astype(np.float32)
        gray, restore = self._center_pad_or_crop_np(gray, self.image_size)
        nonzero = gray != 0
        if np.any(nonzero):
            mean = float(gray[nonzero].mean())
            values = gray[nonzero]
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            gray = (gray - mean) / (std + 1e-8)
        else:
            gray = np.zeros_like(gray, dtype=np.float32)
        x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()  # 1,1,H,W
        return x.to(device), restore

    def _logits_to_masks(self, logits, restore, torch, F):
        h, w = restore["orig_hw"]
        pred_square = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        channel_count = int(logits.shape[1])
        bg_mode = os.environ.get("SPINE_PRED_HAS_BACKGROUND", "auto").lower()
        if self._pred_has_background is not None:
            has_bg = bool(self._pred_has_background)
        elif bg_mode in ("1", "true", "yes"):
            has_bg = True
        elif bg_mode in ("0", "false", "no"):
            has_bg = False
        else:
            has_bg = channel_count == MODEL_TOTAL_CLASSES

        label_square = np.zeros_like(pred_square, dtype=np.uint8)
        if has_bg:
            max_label = min(len(SEG_COLORS_RGB), channel_count - 1)
            for cls_idx in range(1, max_label + 1):
                label_square[pred_square == cls_idx] = cls_idx
        else:
            max_label = min(len(SEG_COLORS_RGB), channel_count)
            for cls_idx in range(max_label):
                label_square[pred_square == cls_idx] = cls_idx + 1

        label_map = np.zeros((h, w), dtype=np.uint8)
        src_y0, src_y1, src_x0, src_x1 = restore["src"]
        dst_y0, dst_y1, dst_x0, dst_x1 = restore["dst"]
        label_map[src_y0:src_y1, src_x0:src_x1] = label_square[dst_y0:dst_y1, dst_x0:dst_x1]

        mask = np.zeros((h, w, 3), dtype=np.uint8)
        for label_id, color in enumerate(SEG_COLORS_RGB, start=1):
            mask[label_map == label_id] = color
        return mask, label_map

    def _label_map_to_display(self, label_map):
        if self.meta.get("infer_display_transform") == "nifti":
            label_map = nifti_slice_to_display(label_map)
        return np.ascontiguousarray(label_map, dtype=np.uint8)

    def run(self):
        try:
            import torch
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            import torch.nn.functional as F

            self.progress.emit(5, "加载模型与权重...")
            model, device = self._load_model(torch, F)

            seg_masks = []
            label_maps = []
            clinical_label_maps = []
            slice_times = []
            n = max(1, len(self.infer_arrays))
            with torch.no_grad():
                for i, arr in enumerate(self.infer_arrays):
                    slice_t0 = time.time()
                    self.progress.emit(28 + int(i / n * 65), f"推理切片 {i+1}/{n}...")
                    x, restore = self._preprocess_slice(arr, torch, F, device)
                    out = model(x)
                    if getattr(device, "type", "") == "cuda":
                        torch.cuda.synchronize(device)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if out.ndim == 3:
                        out = out.unsqueeze(0)
                    _, raw_label_map = self._logits_to_masks(out, restore, torch, F)
                    label_map = self._label_map_to_display(raw_label_map)
                    if self.meta.get("source_format") == "dicom":
                        clinical_label_map = inverse_medical_orientation(raw_label_map)
                    elif self.meta.get("source_format") == "raster":
                        # Raster angle measurements use the visible image axes.
                        # Endplate-to-endplate angles are rotation invariant, while
                        # absolute sacral slope remains gated separately.
                        clinical_label_map = np.ascontiguousarray(label_map, dtype=np.uint8)
                    else:
                        clinical_label_map = np.ascontiguousarray(raw_label_map, dtype=np.uint8)
                    mask = labels_to_color_mask(label_map)
                    seg_masks.append(mask)
                    label_maps.append(label_map)
                    clinical_label_maps.append(clinical_label_map)
                    slice_times.append(time.time() - slice_t0)

            analysis = analyze_segmentation(
                self.orig_arrays,
                label_maps,
                self.meta,
                clinical_label_maps=clinical_label_maps,
            )
            self.progress.emit(100, "分割完成 ✓")
            self.finished.emit({"seg_masks": seg_masks, "label_maps": label_maps, "analysis": analysis, "slice_times": slice_times})
        except Exception:
            self.error.emit(traceback.format_exc())


# ─── Export Dialog ───────────────────────────────────────────────────────────

class ExportDialog(QDialog):
    def __init__(self, parent=None, n_slices=1):
        super().__init__(parent)
        self.setWindowTitle("导出分割结果")
        self.setFixedSize(460, 280)
        self.setStyleSheet(STYLESHEET)
        self.result_fmt = "png_series"
        self.result_path = ""
        self.n_slices = n_slices
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(12); lay.setContentsMargins(20,20,20,20)
        title = QLabel("选择导出格式与路径")
        title.setStyleSheet(f"font-size:14px;font-weight:bold;color:{C['cyan']};")
        lay.addWidget(title)

        fmt_row = QHBoxLayout(); fmt_row.addWidget(QLabel("导出格式"))
        self.fmt_combo = QComboBox()
        opts = ["PNG 序列 (分割掩码)", "PNG 序列 (叠加原图)"]
        if HAS_NIBABEL:
            opts.append("NIfTI 标签体积 (.nii.gz)")
        else:
            opts.append("NIfTI (需要 pip install nibabel)")
        self.fmt_combo.addItems(opts)
        fmt_row.addWidget(self.fmt_combo); lay.addLayout(fmt_row)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(); self.path_edit.setPlaceholderText("选择保存路径...")
        path_row.addWidget(self.path_edit)
        browse_btn = QPushButton("📁 浏览"); browse_btn.setFixedWidth(72)
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C['bg_card']};border:1px solid {C['border']};
                color:{C['cyan']};font-size:12px;padding:4px 8px;
            }}
            QPushButton:hover {{border-color:{C['cyan']};background:rgba(0,212,255,20);}}
        """)
        browse_btn.clicked.connect(self._browse_path)
        path_row.addWidget(browse_btn); lay.addLayout(path_row)

        info = QLabel(f"当前序列共 {self.n_slices} 个切片将被导出。")
        info.setStyleSheet(f"font-size:11px;color:{C['text_mid']};"); info.setWordWrap(True)
        lay.addWidget(info); lay.addStretch()

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.setStyleSheet(f"""
            QPushButton {{
                background:{C['bg_card']};border:1px solid {C['border']};
                color:{C['cyan']};padding:5px 20px;min-width:70px;
            }}
            QPushButton:hover {{border-color:{C['cyan']};background:rgba(0,212,255,20);}}
        """)
        bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse_path(self):
        if self.fmt_combo.currentIndex() == 2:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存 NIfTI", "segmentation.nii.gz",
                "NIfTI (*.nii.gz *.nii);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if path:
            self.path_edit.setText(path)

    def _accept(self):
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "路径为空", "请先选择保存路径。"); return
        self.result_fmt  = {0:"png_series", 1:"orig_overlay_png", 2:"nifti"}.get(
            self.fmt_combo.currentIndex(), "png_series")
        self.result_path = path; self.accept()


# ─── Main Window ────────────────────────────────────────────────────────────

class SpineVisionApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpineVision AI — 脊柱智能分割系统 v2.4")
        self.resize(1500, 880)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLESHEET)

        # State
        self._series = []           # list of dict
        self._active_idx = -1
        self._seg_alpha = 0.55
        self._model_path = resolve_default_model_path()
        self._orig_arrays = []
        self._infer_arrays = []
        self._seg_arrays = []
        self._label_arrays = []
        self._analysis = None
        self._slice_times = []
        self._total_infer_time = 0.0

        self._build_ui()
        self._setup_shortcuts()

    # ═══════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._build_topbar())
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{C['border']};border:none;"); root.addWidget(sep)

        content = QHBoxLayout(); content.setContentsMargins(0,0,0,0); content.setSpacing(1)

        def vsep():
            f = QFrame(); f.setFrameShape(QFrame.VLine)
            f.setStyleSheet(f"background:{C['border']};border:none;"); f.setFixedWidth(1)
            return f

        content.addWidget(self._build_left_sidebar())
        content.addWidget(vsep())
        content.addWidget(self._build_center_panels(), stretch=1)
        content.addWidget(vsep())
        content.addWidget(self._build_right_sidebar())
        root.addLayout(content, stretch=1)

    # ── Top bar ──────────────────────────────────────────────────────────

    def _build_topbar(self):
        wrapper = QWidget(); wrapper.setFixedHeight(52)
        TopBar(wrapper).setGeometry(0, 0, 9999, 52)
        lay = QHBoxLayout(wrapper); lay.setContentsMargins(20,0,20,0); lay.setSpacing(14)

        icon = QLabel("⬡")
        icon.setStyleSheet(f"font-size:26px;color:{C['cyan']};background:transparent;")
        lay.addWidget(icon)
        vb = QVBoxLayout(); vb.setSpacing(0)
        vb.addWidget(self._styled_label("SPINEVISION AI",
            f"font-family:'Rajdhani','Microsoft YaHei';font-size:18px;font-weight:bold;"
            f"color:{C['cyan']};letter-spacing:3px;background:transparent;"))
        vb.addWidget(self._styled_label("脊柱智能分割系统  v2.4",
            f"font-size:10px;color:{C['text_mid']};letter-spacing:2px;background:transparent;"))
        lay.addLayout(vb); lay.addStretch()

        self.status_lbl = QLabel("● MODEL READY")
        self._style_status_ready()
        lay.addWidget(self.status_lbl)

        for lbl, slot in [("导入影像", self._action_import),
                          ("导出结果", self._action_export),
                          ("结构量化", self._action_quantify),
                          ("系统设置", self._action_settings)]:
            btn = CyberButton(lbl); btn.clicked.connect(slot); lay.addWidget(btn)
        return wrapper

    # ── Left sidebar ─────────────────────────────────────────────────────

    def _build_left_sidebar(self):
        w = QWidget(); w.setFixedWidth(252)
        w.setStyleSheet(f"background:{C['bg_panel']};")
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        lay.addWidget(SectionHeader("影像序列"))

        # Drop-zone import button
        import_btn = QPushButton(
            "🩻  拖拽或点击导入\nDICOM / NIfTI / PNG / JPG\n.dcm  .nii  .nii.gz  .png  .jpg")
        import_btn.setFixedHeight(80); import_btn.setCursor(Qt.PointingHandCursor)
        import_btn.setStyleSheet(f"""
            QPushButton {{
                background:rgba(0,100,160,15); border:1px dashed rgba(0,180,220,80);
                color:{C['text_mid']}; font-size:11px; padding:8px;
                text-align:center; margin:10px;
            }}
            QPushButton:hover {{
                background:rgba(0,180,220,20); border-color:{C['cyan']};
                color:{C['text_bright']};
            }}
        """)
        import_btn.clicked.connect(self._action_import)
        lay.addWidget(import_btn)

        # DICOM folder import
        folder_btn = QPushButton("📂  导入 DICOM 文件夹")
        folder_btn.setFixedHeight(28); folder_btn.setCursor(Qt.PointingHandCursor)
        folder_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; border:1px solid {C['border']};
                color:{C['text_mid']}; font-size:11px; margin:0 10px 6px 10px;
            }}
            QPushButton:hover {{ border-color:{C['cyan_dim']}; color:{C['cyan']}; }}
        """)
        folder_btn.clicked.connect(self._action_import_folder)
        lay.addWidget(folder_btn)

        # Series list
        self.series_list_widget = QListWidget()
        self.series_list_widget.setFixedHeight(130)
        self.series_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.series_list_widget.customContextMenuRequested.connect(self._series_context_menu)
        self.series_list_widget.currentRowChanged.connect(self._on_series_selected)
        lay.addWidget(self.series_list_widget)

        hdiv = QFrame(); hdiv.setFrameShape(QFrame.HLine)
        hdiv.setStyleSheet(f"background:{C['border']};border:none;"); hdiv.setFixedHeight(1)
        lay.addWidget(hdiv); lay.addWidget(SectionHeader("模型参数"))

        # Model selector + browse button
        lay.addWidget(self._styled_label("分割模型",
            f"font-size:11px;color:{C['text_mid']};padding:8px 14px 2px 14px;background:transparent;"))
        model_row = QHBoxLayout(); model_row.setContentsMargins(10,0,10,0); model_row.setSpacing(4)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "SwinUMamba-D (内置)", "nnU-Net v2 (内置)",
            "TransUNet (内置)", "Swin-UNet (内置)",
        ])
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_row.addWidget(self.model_combo, stretch=1)
        browse_model_btn = QPushButton("📁")
        browse_model_btn.setFixedSize(28, 28)
        browse_model_btn.setToolTip("从文件导入模型权重 (.pth/.pt/.onnx)")
        browse_model_btn.setCursor(Qt.PointingHandCursor)
        browse_model_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C['bg_card']}; border:1px solid {C['border']};
                color:{C['cyan']}; font-size:14px;
            }}
            QPushButton:hover {{border-color:{C['cyan']};background:rgba(0,212,255,20);}}
        """)
        browse_model_btn.clicked.connect(self._browse_model_file)
        model_row.addWidget(browse_model_btn); lay.addLayout(model_row)

        self.model_path_lbl = QLabel(short_path_label(self._model_path))
        self.model_path_lbl.setStyleSheet(
            f"font-family:'Courier New';font-size:9px;color:{C['text_dim']};"
            f"padding:0 14px 4px 14px;background:transparent;")
        self.model_path_lbl.setWordWrap(True)
        lay.addWidget(self.model_path_lbl)

        # Other combos
        for label, options in [
            ("分割目标", ["背景 + 12 个前景 mask", "仅椎体 L1-S5", "仅椎管", "全脊柱"]),
            ("计算设备", ["GPU — NVIDIA RTX 4090", "GPU — NVIDIA A100", "CPU (慢速)"]),
        ]:
            lay.addWidget(self._styled_label(label,
                f"font-size:11px;color:{C['text_mid']};padding:6px 14px 2px 14px;background:transparent;"))
            cb = QComboBox(); cb.addItems(options); cb.setFixedHeight(28)
            cb.setStyleSheet(cb.styleSheet() + "margin:0 10px;")
            lay.addWidget(cb)

        # Sliders
        for s_label, s_val in [("置信度阈值", 65), ("后处理强度", 40)]:
            lay.addWidget(self._styled_label(s_label,
                f"font-size:11px;color:{C['text_mid']};padding:6px 14px 2px 14px;background:transparent;"))
            row = QHBoxLayout(); row.setContentsMargins(10,0,10,0)
            sl = QSlider(Qt.Horizontal); sl.setRange(0, 100); sl.setValue(s_val)
            vl = QLabel(f"{s_val}%"); vl.setFixedWidth(34)
            vl.setStyleSheet(f"font-family:'Courier New';font-size:11px;color:{C['cyan']};background:transparent;")
            sl.valueChanged.connect(lambda v, l=vl: l.setText(f"{v}%"))
            row.addWidget(sl); row.addWidget(vl); lay.addLayout(row)

        lay.addStretch()
        return w

    # ── Center panels ─────────────────────────────────────────────────────

    def _build_center_panels(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(1)
        row = QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.setSpacing(1)
        row.addWidget(self._build_image_panel("脊柱原图",       ["⊹","⊕","✥","⊛","↺"], "orig"))
        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"background:{C['border']};border:none;"); sep.setFixedWidth(1)
        row.addWidget(sep)
        row.addWidget(self._build_image_panel("脊柱分割后效果图",["◈","▦","□","◑","◉"], "seg"))
        lay.addLayout(row, stretch=1)
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"background:{C['border']};border:none;"); sep2.setFixedHeight(1)
        lay.addWidget(sep2); lay.addWidget(self._build_info_bar())
        return w

    def _build_image_panel(self, title, tools, name):
        w = QWidget(); w.setStyleSheet(f"background:{C['bg_panel']};")
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        header = QWidget(); header.setFixedHeight(36)
        header.setStyleSheet(f"background:rgba(0,6,20,180);border-bottom:1px solid {C['border']};")
        hlay = QHBoxLayout(header); hlay.setContentsMargins(12,0,8,0)
        hlay.addWidget(self._styled_label(title,
            f"font-family:'Rajdhani','Microsoft YaHei';font-size:13px;font-weight:bold;"
            f"color:{C['cyan']};letter-spacing:2px;background:transparent;"))
        hlay.addStretch()
        for t in tools:
            b = QPushButton(t); b.setFixedSize(26,26); b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(f"""
                QPushButton {{
                    background:transparent;border:1px solid {C['border']};
                    color:{C['text_mid']};font-size:13px;
                }}
                QPushButton:hover {{
                    background:rgba(0,212,255,30);border-color:{C['cyan']};color:{C['cyan']};
                }}
            """)
            hlay.addWidget(b)
        lay.addWidget(header)

        if name == "orig":
            self.orig_viewport = ImageViewport("🦴", "暂无影像数据\n请从左侧导入序列")
            self.orig_viewport.slice_changed.connect(self._on_orig_slice_wheel)
            lay.addWidget(self.orig_viewport, stretch=1)
        else:
            self.seg_viewport = ImageViewport("🔬", "等待分割完成\n点击右侧按钮开始")
            self.seg_viewport.slice_changed.connect(self._on_seg_slice_wheel)
            lay.addWidget(self.seg_viewport, stretch=1)

        # Slice bar
        sbar = QWidget(); sbar.setFixedHeight(34)
        sbar.setStyleSheet(f"background:rgba(0,4,14,180);border-top:1px solid {C['border']};")
        slay = QHBoxLayout(sbar); slay.setContentsMargins(12,0,12,0); slay.setSpacing(8)
        slay.addWidget(self._mono_label("切片"))
        sl = QSlider(Qt.Horizontal); sl.setRange(1,64); sl.setValue(1); slay.addWidget(sl)
        slbl = QLabel("1 / 1"); slbl.setFixedWidth(52)
        slbl.setStyleSheet(f"font-family:'Courier New';font-size:11px;color:{C['cyan']};background:transparent;")
        slay.addWidget(slbl)

        if name == "orig":
            self.orig_slice_slider = sl; self.orig_slice_lbl = slbl
            sl.valueChanged.connect(self._on_orig_slice_slider)
        else:
            self.seg_slice_slider = sl; self.seg_slice_lbl = slbl
            sl.valueChanged.connect(self._on_seg_slice_slider)
            slay.addWidget(self._mono_label("透明度"))
            al = QSlider(Qt.Horizontal); al.setRange(0,100); al.setValue(55); al.setFixedWidth(80)
            slay.addWidget(al)
            al_lbl = QLabel("55%"); al_lbl.setFixedWidth(34)
            al_lbl.setStyleSheet(f"font-family:'Courier New';font-size:11px;color:{C['cyan']};background:transparent;")
            al.valueChanged.connect(lambda v, l=al_lbl: (l.setText(f"{v}%"), self._set_alpha(v)))
            slay.addWidget(al_lbl)
            self.alpha_slider = al

        lay.addWidget(sbar)
        return w

    def _build_info_bar(self):
        w = QWidget(); w.setFixedHeight(32); w.setStyleSheet(f"background:{C['bg_panel']};")
        lay = QHBoxLayout(w); lay.setContentsMargins(16,0,16,0); lay.setSpacing(28)
        self._info_labels = {}
        for key, val in [("患者","—"),("序列","—"),("尺寸","—"),("间距","—"),("模态","—")]:
            pair = QHBoxLayout(); pair.setSpacing(6)
            pair.addWidget(self._dim_label(key))
            vl = self._mono_label(val, C['cyan']); self._info_labels[key] = vl
            pair.addWidget(vl); lay.addLayout(pair)
        lay.addStretch()
        self.infer_time_lbl = self._mono_label("推理时间: —", C['text_mid'])
        lay.addWidget(self.infer_time_lbl)
        return w

    # ── Right sidebar ─────────────────────────────────────────────────────

    def _build_right_sidebar(self):
        w = QWidget(); w.setFixedWidth(320); w.setStyleSheet(f"background:{C['bg_panel']};")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border:none;background:transparent;")
        inner = QWidget(); inner.setStyleSheet(f"background:{C['bg_panel']};")
        lay = QVBoxLayout(inner); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # Class list
        lay.addWidget(SectionHeader("分割类别"))
        cls_list = QListWidget(); cls_list.setStyleSheet("background:transparent;border:none;")
        cls_list.setFixedHeight(min(len(SEG_CLASSES)*28, 300))
        for name, color, vol in SEG_CLASSES:
            item = QListWidgetItem(); item.setText(f"  {name}")
            item.setForeground(QColor(C['text_bright'])); item.setIcon(self._color_icon(color))
            cls_list.addItem(item)
        lay.addWidget(cls_list)

        self._hdiv(lay)

        # Metrics
        lay.addWidget(SectionHeader("量化指标"))
        mw = QWidget(); mw.setStyleSheet("background:transparent;")
        mw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        mvlay = QVBoxLayout(mw); mvlay.setContentsMargins(10,8,10,8); mvlay.setSpacing(6)
        self.metric_cards = []
        md = [("—","/12","检测到的前景类别","等待推理",True),("—","%","前景像素占比","等待推理",True),
              ("—","°","L1–S1 腰椎前凸角","等待推理",True),("—","","椎管 mask 量化","等待推理",True)]
        for ri in range(0, len(md), 2):
            rw = QWidget(); rw.setStyleSheet("background:transparent;")
            rw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            rh = QHBoxLayout(rw); rh.setContentsMargins(0,0,0,0); rh.setSpacing(6)
            for ci in range(2):
                idx = ri+ci
                if idx < len(md):
                    card = MetricCard(*md[idx]); card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                    self.metric_cards.append(card)
                    rh.addWidget(card, stretch=1)
            mvlay.addWidget(rw)
        lay.addWidget(mw); self._hdiv(lay)

        lay.addWidget(SectionHeader("标准化测量与可用性")); lay.addWidget(self._build_diagnosis_card())
        self._hdiv(lay)
        lay.addWidget(SectionHeader("复核与限制")); lay.addWidget(self._build_treatment_card())
        lay.addStretch(); scroll.setWidget(inner)

        outer = QVBoxLayout(w); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        outer.addWidget(scroll, stretch=1)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{C['border']};border:none;"); sep.setFixedHeight(1)
        outer.addWidget(sep)

        btn_area = QWidget(); btn_area.setStyleSheet(f"background:{C['bg_panel']};")
        blay = QVBoxLayout(btn_area); blay.setContentsMargins(12,10,12,12); blay.setSpacing(6)

        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(4); self.progress_bar.setTextVisible(False)
        self.progress_bar.hide(); blay.addWidget(self.progress_bar)

        self.progress_lbl = QLabel("")
        self.progress_lbl.setStyleSheet(
            f"font-family:'Courier New';font-size:10px;color:{C['cyan']};background:transparent;")
        self.progress_lbl.hide(); blay.addWidget(self.progress_lbl)

        self.seg_btn = CyberButton("⚡  开始分割", accent=True)
        self.seg_btn.setFixedHeight(42); self.seg_btn.clicked.connect(self._action_segment)
        blay.addWidget(self.seg_btn); outer.addWidget(btn_area)
        return w

    def _build_diagnosis_card(self):
        w = QFrame(); w.setStyleSheet(f"""
            QFrame {{
                background:{C['bg_card']};border:1px solid {C['border']};
                border-left:3px solid {C['cyan']};margin:0 10px 8px 10px;
            }}
        """)
        lay = QVBoxLayout(w); lay.setContentsMargins(12,10,12,10); lay.setSpacing(4)
        lay.addWidget(self._styled_label("临床指标（研究辅助）",
            f"font-family:'Rajdhani','Microsoft YaHei';font-size:13px;font-weight:bold;"
            f"color:{C['cyan']};background:transparent;border:none;"))
        self.diagnosis_value_labels = []
        for key, val, warn in [
            ("L1–S1 腰椎前凸角","等待分割",False),("L4–S1 下腰椎前凸角","等待分割",False),
            ("S1 骶骨倾斜角","等待分割",False),("椎管 mask 量化","等待分割",False),
            ("椎间盘高度/退变","等待分割",False),("椎管狭窄分级","等待分割",False),
            ("骨密度/骨质疏松","等待分割",False),("椎体压缩性骨折","等待分割",False),
        ]:
            row = QHBoxLayout(); row.setContentsMargins(0,1,0,1)
            k = QLabel(key); k.setStyleSheet(f"font-size:11px;color:{C['text_mid']};background:transparent;border:none;")
            v = QLabel(val); v.setStyleSheet(
                f"font-family:'Courier New';font-size:11px;"
                f"color:{C['orange'] if warn else '#00ff88'};background:transparent;border:none;")
            v.setWordWrap(True)
            v.setMaximumWidth(190)
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.diagnosis_value_labels.append((k, v))
            row.addWidget(k); row.addStretch(); row.addWidget(v); lay.addLayout(row)
        return w

    def _build_treatment_card(self):
        w = QFrame(); w.setStyleSheet(f"""
            QFrame {{
                background:rgba(0,80,120,20);
                border:1px solid rgba(0,212,255,40);margin:0 10px 8px 10px;
            }}
        """)
        lay = QVBoxLayout(w); lay.setContentsMargins(12,10,12,10); lay.setSpacing(5)
        lay.addWidget(self._styled_label("⚕  测量边界",
            f"font-size:13px;font-weight:bold;color:{C['teal']};background:transparent;border:none;"))
        self.treatment_labels = []
        for item in ["等待分割后检查各项指标的输入条件",
                     "本页仅显示结构化测量，不自动生成临床诊断或治疗方案",
                     "所有自动测量均需由影像科或脊柱专科医师复核",
                     "条件不足的指标将明确显示“不可计算”"]:
            row = QHBoxLayout(); row.setContentsMargins(0,0,0,0)
            arrow = QLabel("▸"); arrow.setFixedWidth(14)
            arrow.setStyleSheet(f"color:{C['teal']};background:transparent;border:none;font-size:11px;")
            txt = QLabel(item); txt.setWordWrap(True)
            txt.setStyleSheet(f"font-size:11px;color:{C['text_mid']};background:transparent;border:none;")
            self.treatment_labels.append(txt)
            row.addWidget(arrow, 0, Qt.AlignTop); row.addWidget(txt); lay.addLayout(row)
        return w

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _styled_label(self, text, style):
        l = QLabel(text); l.setStyleSheet(style); return l

    def _mono_label(self, text, color=None):
        return self._styled_label(text,
            f"font-family:'Courier New';font-size:11px;"
            f"color:{color or C['text_mid']};background:transparent;")

    def _dim_label(self, text):
        return self._styled_label(text,
            f"font-size:11px;color:{C['text_dim']};background:transparent;")

    def _color_icon(self, hex_color):
        px = QPixmap(12,12); px.fill(QColor(hex_color)); return QIcon(px)

    def _hdiv(self, lay):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"background:{C['border']};border:none;margin:0 12px;")
        f.setFixedHeight(1); lay.addWidget(f)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, self._action_import)
        QShortcut(QKeySequence("Ctrl+S"), self, self._action_export)
        QShortcut(QKeySequence("Ctrl+R"), self, self._action_segment)

    def _style_status_ready(self):
        self.status_lbl.setStyleSheet(
            f"font-family:'Courier New';font-size:11px;color:{C['green']};"
            f"background:rgba(0,255,100,15);border:1px solid rgba(0,255,100,60);"
            f"border-radius:10px;padding:4px 12px;")

    def _style_status_busy(self, text="● 处理中..."):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(
            f"font-family:'Courier New';font-size:11px;color:{C['orange']};"
            f"background:rgba(255,107,43,10);border:1px solid rgba(255,107,43,50);"
            f"border-radius:10px;padding:4px 12px;")

    # ═══════════════════════════════════════════════════════════════════
    #  SERIES MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _add_series(self, name, meta, orig_arrays):
        infer_arrays = meta.pop("infer_arrays", None) if isinstance(meta, dict) else None
        if infer_arrays is None:
            infer_arrays = orig_arrays
        entry = {"name": name, "meta": meta,
                 "orig_arrays": orig_arrays, "infer_arrays": infer_arrays,
                 "seg_arrays": [],
                 "label_arrays": [], "analysis": None,
                 "slice_times": [], "total_infer_time": 0.0}
        self._series.append(entry)
        n = len(orig_arrays)
        item = QListWidgetItem()
        item.setText(f"  🗂  {name}\n"
                     f"     {meta.get('shape','—')}  {meta.get('modality','—')}")
        item.setForeground(QColor(C['text_bright']))
        self.series_list_widget.addItem(item)
        self.series_list_widget.setCurrentRow(len(self._series) - 1)

    def _on_series_selected(self, row):
        if row < 0 or row >= len(self._series): return
        self._active_idx = row
        entry = self._series[row]
        self._orig_arrays = entry["orig_arrays"]
        self._infer_arrays = entry.get("infer_arrays")
        if self._infer_arrays is None:
            self._infer_arrays = self._orig_arrays
        self._seg_arrays  = entry["seg_arrays"]
        self._label_arrays = entry.get("label_arrays", [])
        self._analysis = entry.get("analysis")
        self._slice_times = entry.get("slice_times", [])
        self._total_infer_time = float(entry.get("total_infer_time", 0.0) or 0.0)
        self._load_orig_to_viewport()
        self._load_seg_to_viewport()
        self._update_info_bar(entry["meta"])
        self._update_analysis_panel(self._analysis)
        self._update_infer_time_label()

    def _series_context_menu(self, pos):
        item = self.series_list_widget.itemAt(pos)
        if item is None: return
        menu = QMenu(self)
        act = QAction("🗑  移除此序列", self)
        act.triggered.connect(self._remove_current_series)
        menu.addAction(act); menu.exec_(self.series_list_widget.mapToGlobal(pos))

    def _remove_current_series(self):
        row = self.series_list_widget.currentRow()
        if row < 0: return
        self._series.pop(row)
        self.series_list_widget.takeItem(row)
        if self._series:
            self.series_list_widget.setCurrentRow(min(row, len(self._series)-1))
        else:
            self._active_idx = -1; self._orig_arrays = []; self._infer_arrays = []; self._seg_arrays = []
            self._label_arrays = []; self._analysis = None
            self._slice_times = []; self._total_infer_time = 0.0
            self.orig_viewport.set_slices([]); self.seg_viewport.set_slices([])
            for lbl in self._info_labels.values(): lbl.setText("—")
            self._update_analysis_panel(None)
            self._update_infer_time_label()

    # ═══════════════════════════════════════════════════════════════════
    #  VIEWPORT / SLICE SYNC
    # ═══════════════════════════════════════════════════════════════════

    def _update_infer_time_label(self, idx=None):
        if not hasattr(self, "infer_time_lbl"):
            return
        times = list(self._slice_times or [])
        total = float(self._total_infer_time or 0.0)
        if idx is None:
            if hasattr(self, "orig_slice_slider"):
                idx = max(0, self.orig_slice_slider.value() - 1)
            else:
                idx = 0
        if not times:
            self.infer_time_lbl.setText(f"总推理时间: {total:.2f}s" if total > 0 else "推理时间: —")
            return
        idx = max(0, min(int(idx), len(times) - 1))
        current = float(times[idx])
        avg = float(sum(times) / len(times))
        if len(times) > 1:
            self.infer_time_lbl.setText(f"总时间: {total:.2f}s | 当前: {current:.2f}s | 平均: {avg:.2f}s")
        else:
            self.infer_time_lbl.setText(f"推理时间: {total:.2f}s | 当前: {current:.2f}s")

    def _load_orig_to_viewport(self):
        pms = [numpy_to_pixmap(a) for a in self._orig_arrays]
        self.orig_viewport.set_slices(pms)
        n = max(len(pms), 1)
        self.orig_slice_slider.setRange(1, n); self.orig_slice_slider.setValue(1)
        self.orig_slice_lbl.setText(f"1 / {n}")

    def _load_seg_to_viewport(self):
        if not self._seg_arrays:
            self.seg_viewport.set_slices([]); return
        alpha = self._seg_alpha
        blended = [numpy_to_pixmap(blend_overlay(o, s, alpha))
                   for o, s in zip(self._orig_arrays, self._seg_arrays)]
        self.seg_viewport.set_slices(blended)
        n = max(len(blended), 1)
        self.seg_slice_slider.setRange(1, n); self.seg_slice_slider.setValue(1)
        self.seg_slice_lbl.setText(f"1 / {n}")

    def _set_slider_without_signal(self, slider, value):
        old = slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(old)

    def _on_orig_slice_slider(self, val):
        self.orig_viewport.set_slice_index(val - 1)
        self.orig_slice_lbl.setText(f"{val} / {self.orig_slice_slider.maximum()}")
        self._update_infer_time_label(val - 1)
        if self._seg_arrays and self.seg_slice_slider.maximum() >= val:
            self.seg_viewport.set_slice_index(val - 1)
            self._set_slider_without_signal(self.seg_slice_slider, val)
            self.seg_slice_lbl.setText(f"{val} / {self.seg_slice_slider.maximum()}")

    def _on_seg_slice_slider(self, val):
        self.seg_viewport.set_slice_index(val - 1)
        self.seg_slice_lbl.setText(f"{val} / {self.seg_slice_slider.maximum()}")
        self._update_infer_time_label(val - 1)
        if self._orig_arrays and self.orig_slice_slider.maximum() >= val:
            self.orig_viewport.set_slice_index(val - 1)
            self._set_slider_without_signal(self.orig_slice_slider, val)
            self.orig_slice_lbl.setText(f"{val} / {self.orig_slice_slider.maximum()}")

    def _on_orig_slice_wheel(self, idx): self.orig_slice_slider.setValue(idx + 1)
    def _on_seg_slice_wheel(self, idx):  self.seg_slice_slider.setValue(idx + 1)

    def _set_alpha(self, v):
        self._seg_alpha = v / 100.0
        if self._seg_arrays and self._orig_arrays:
            self._load_seg_to_viewport()

    def _update_info_bar(self, meta):
        modality = meta.get("modality", "—")
        input_domain = meta.get("model_input_domain")
        if input_domain:
            modality = f"{modality} · {input_domain}"
        mapping = {
            "患者": meta.get("filename", "—"),
            "序列": meta.get("filename", "—"),
            "尺寸": meta.get("shape", "—"),
            "间距": meta.get("spacing", "—"),
            "模态": modality,
        }
        for k, v in mapping.items():
            if k in self._info_labels:
                self._info_labels[k].setText(v)

    def _update_analysis_panel(self, analysis):
        if not hasattr(self, "diagnosis_value_labels"):
            return
        if not analysis:
            for i, (k, v) in enumerate(self.diagnosis_value_labels):
                v.setText("等待分割")
                v.setStyleSheet(f"font-family:'Courier New';font-size:11px;color:#00ff88;background:transparent;border:none;")
            if hasattr(self, "metric_cards"):
                defaults = [("—","/12","检测到的前景类别","等待推理",True),
                            ("—","%","前景像素占比","等待推理",True),
                            ("—","°","L1–S1 腰椎前凸角","等待推理",True),
                            ("—","","椎管 mask 量化","等待推理",True)]
                for card, data in zip(self.metric_cards, defaults):
                    card.set_values(*data)
            for lbl, text in zip(getattr(self, "treatment_labels", []), [
                "等待分割后检查各项指标的输入条件",
                "本页仅显示结构化测量，不自动生成临床诊断或治疗方案",
                "所有自动测量均需由影像科或脊柱专科医师复核",
                "条件不足的指标将明确显示“不可计算”",
            ]):
                lbl.setText(text)
            return

        rows = analysis.get("rows", [])
        for i, (k, v) in enumerate(self.diagnosis_value_labels):
            if i >= len(rows):
                continue
            row = rows[i]
            k.setText(row.get("key", "—"))
            warn = bool(row.get("warn"))
            value_text = row.get("value", "—")
            v.setText(value_text)
            v.setToolTip(
                f"{value_text}\n方法：{row.get('method', '—')}\n"
                f"状态：{row.get('status', '—')}"
            )
            v.setStyleSheet(
                f"font-family:'Courier New';font-size:11px;"
                f"color:{C['orange'] if warn else '#00ff88'};background:transparent;border:none;")

        for card, item in zip(getattr(self, "metric_cards", []), analysis.get("summary_metrics", [])):
            card.set_values(item.get("value", "—"), item.get("unit", ""),
                            item.get("label", "—"), item.get("trend", ""),
                            bool(item.get("ok", True)))

        for lbl, text in zip(getattr(self, "treatment_labels", []), analysis.get("recommendations", [])):
            lbl.setText(text)

    # ═══════════════════════════════════════════════════════════════════
    #  ACTIONS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_plain_image_path(path):
        return Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def _add_png_batch(self, paths):
        slices = []
        infer_arrays = []
        names = []
        first_meta = None
        failed = []
        for path in paths:
            loaded, meta = load_image_file(path)
            if loaded is None:
                failed.append(str(path))
                continue
            names.append(Path(path).name)
            slices.extend(loaded)
            infer = meta.get("infer_arrays")
            if infer is None:
                infer = [rgb_to_gray_float(a) for a in loaded]
            infer_arrays.extend(infer)
            if first_meta is None:
                first_meta = {k: v for k, v in meta.items() if k != "infer_arrays"}

        if not slices:
            return False

        h, w = slices[0].shape[:2]
        meta = dict(first_meta or {})
        meta.update({
            "path": str(Path(paths[0]).parent),
            "filename": f"{len(slices)} PNG images",
            "modality": "PNG_BATCH",
            "shape": f"{w}x{h}x{len(slices)}",
            "spacing": meta.get("spacing", "--"),
            "file_names": names,
            "infer_arrays": infer_arrays,
        })
        self._add_series(f"PNG batch ({len(slices)})", meta, slices)
        if failed:
            QMessageBox.warning(self, "部分图片读取失败", "\n".join(failed[:8]))
        return True

    def _action_import(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入医学影像", "",
            "医学影像 (*.dcm *.nii *.nii.gz *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
            "NIfTI (*.nii *.nii.gz);;DICOM (*.dcm);;"
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif);;所有文件 (*)"
        )
        if not paths: return
        if len(paths) > 1:
            if not all(self._is_plain_image_path(p) for p in paths):
                QMessageBox.warning(self, "批量导入限制", "多选批量导入仅支持 PNG/JPG/BMP/TIF 图片。")
                return
            self._style_status_busy("● 读取批量图片...")
            QApplication.processEvents()
            if self._add_png_batch(paths):
                self.status_lbl.setText("● MODEL READY")
                self._style_status_ready()
            else:
                self.status_lbl.setText("● 读取失败")
            return
        path = paths[0]
        self._style_status_busy("● 读取文件...")
        QApplication.processEvents()
        slices, meta = load_image_file(path)
        if slices is None:
            self.status_lbl.setText("● 读取失败"); return
        self._add_series(Path(path).name, meta, slices)
        self.status_lbl.setText("● MODEL READY"); self._style_status_ready()

    def _action_import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 DICOM 文件夹", "")
        if not folder: return
        self._style_status_busy("● 读取文件夹...")
        QApplication.processEvents()
        slices, meta = load_dicom_folder(folder)
        if slices is None:
            self.status_lbl.setText("● 读取失败"); return
        self._add_series(Path(folder).name, meta, slices)
        self._style_status_ready()

    def _browse_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重文件", "",
            "模型权重 (*.pth *.pt *.onnx *.pkl *.bin);;所有文件 (*)"
        )
        if not path: return
        self._model_path = path
        self.model_path_lbl.setText(short_path_label(path))
        self.model_combo.insertItem(0, f"自定义: {Path(path).stem}")
        self.model_combo.setCurrentIndex(0)
        QMessageBox.information(
            self, "模型权重已加载",
            f"已选择权重文件:\n{path}\n\n"
            f"点击「开始分割」会使用该权重执行真实 PyTorch 推理。\n"
            f"模型结构文件请放在程序同级目录，或设置 SPINE_MODEL_PY。"
        )

    def _action_export(self):
        if not self._seg_arrays:
            QMessageBox.warning(self, "无分割结果", "请先执行分割后再导出。"); return
        dlg = ExportDialog(self, len(self._seg_arrays))
        if dlg.exec_() != QDialog.Accepted: return
        fmt, out_path = dlg.result_fmt, dlg.result_path
        if not out_path: return
        try:
            self._style_status_busy("● 导出中...")
            QApplication.processEvents()
            if   fmt == "png_series":      self._export_png_series(out_path)
            elif fmt == "orig_overlay_png":self._export_overlay_png(out_path)
            elif fmt == "nifti":           self._export_nifti(out_path)
            self._style_status_ready()
            QMessageBox.information(self, "导出成功", f"分割结果已保存至:\n{out_path}")
        except Exception as e:
            self._style_status_ready()
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_png_series(self, folder):
        os.makedirs(folder, exist_ok=True)
        for i, arr in enumerate(self._seg_arrays):
            numpy_to_pixmap(arr).save(os.path.join(folder, f"seg_{i+1:04d}.png"))

    def _export_overlay_png(self, folder):
        os.makedirs(folder, exist_ok=True)
        for i, (orig, seg) in enumerate(zip(self._orig_arrays, self._seg_arrays)):
            pm = numpy_to_pixmap(blend_overlay(orig, seg, self._seg_alpha))
            pm.save(os.path.join(folder, f"overlay_{i+1:04d}.png"))

    def _export_nifti(self, save_path):
        if not HAS_NIBABEL:
            raise RuntimeError("导出 NIfTI 需要安装 nibabel:\n  pip install nibabel")
        h, w = self._seg_arrays[0].shape[:2]
        n = len(self._seg_arrays)
        vol = np.zeros((w, h, n), dtype=np.uint8)
        labels = self._label_arrays or [label_map_from_color_mask(a) for a in self._seg_arrays]
        for z, label_img in enumerate(labels):
            vol[:, :, z] = label_img.T
        nib.save(nib.Nifti1Image(vol, np.eye(4)), save_path)

    def _action_segment(self):
        if not self._orig_arrays:
            QMessageBox.warning(self, "无影像数据", "请先导入影像数据再执行分割。"); return
        self.seg_btn.setEnabled(False); self.seg_btn.setText("⏳  分割中...")
        self.progress_bar.setValue(0); self.progress_bar.show()
        self.progress_lbl.setText("初始化..."); self.progress_lbl.show()
        self._style_status_busy("● 分割中...")
        self._t0 = time.time()
        entry = self._series[self._active_idx] if self._active_idx >= 0 else {}
        meta = entry.get("meta", {})
        infer_arrays = entry.get("infer_arrays")
        if infer_arrays is None:
            infer_arrays = self._infer_arrays if self._infer_arrays else self._orig_arrays
        self._worker = SegWorker(self._orig_arrays, self._model_path, meta=meta, infer_arrays=infer_arrays)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_seg_done)
        self._worker.error.connect(self._on_seg_error)
        self._worker.start()

    def _on_progress(self, pct, text):
        self.progress_bar.setValue(pct); self.progress_lbl.setText(text)

    def _on_seg_error(self, msg):
        self.seg_btn.setEnabled(True); self.seg_btn.setText("⚡  开始分割")
        self.progress_lbl.setText("分割失败 ✗")
        self._style_status_ready()
        QMessageBox.critical(self, "分割失败", msg)

    def _on_seg_done(self, result):
        elapsed = time.time() - self._t0
        self._total_infer_time = elapsed
        self.seg_btn.setEnabled(True); self.seg_btn.setText("✓  重新分割")
        if isinstance(result, dict):
            seg_masks = result.get("seg_masks", [])
            label_maps = result.get("label_maps", [])
            analysis = result.get("analysis")
            slice_times = result.get("slice_times", [])
        else:
            seg_masks = result
            label_maps = [label_map_from_color_mask(m) for m in seg_masks]
            meta = self._series[self._active_idx]["meta"] if self._active_idx >= 0 else {}
            analysis = analyze_segmentation(self._orig_arrays, label_maps, meta)
            slice_times = []
        self._seg_arrays = seg_masks
        self._label_arrays = label_maps
        self._analysis = analysis
        self._slice_times = slice_times
        if self._active_idx >= 0:
            self._series[self._active_idx]["seg_arrays"] = seg_masks
            self._series[self._active_idx]["label_arrays"] = label_maps
            self._series[self._active_idx]["analysis"] = analysis
            self._series[self._active_idx]["slice_times"] = slice_times
            self._series[self._active_idx]["total_infer_time"] = elapsed
        self._load_seg_to_viewport()
        self._update_infer_time_label(0)
        self._update_analysis_panel(analysis)
        self.progress_lbl.setText("分割完成 ✓")
        self._style_status_ready()

    def _action_quantify(self):
        if not self._seg_arrays:
            QMessageBox.information(self, "结构量化", "请先执行分割后再进行量化。"); return
        if not self._analysis:
            meta = self._series[self._active_idx]["meta"] if self._active_idx >= 0 else {}
            labels = self._label_arrays or [label_map_from_color_mask(m) for m in self._seg_arrays]
            self._analysis = analyze_segmentation(self._orig_arrays, labels, meta)
        unit_title = "体积/面积" if self._analysis.get("spacing_mode") != "pixel" else "像素数"
        lines = [f"类别             {unit_title:<12} 占比"]
        lines.append("-" * 40)
        for row in self._analysis.get("class_rows", []):
            if row.get("pixels", 0) <= 0:
                continue
            lines.append(f"{row['name']:<16}  {row['measure']:>10}  {row['ratio']*100:>5.1f}%")
        if len(lines) == 2:
            lines.append("未检测到有效分割结构，请复核模型输出。")
        dlg = QDialog(self); dlg.setWindowTitle("结构量化报告")
        dlg.resize(400, 420); dlg.setStyleSheet(STYLESHEET)
        lay = QVBoxLayout(dlg)
        lay.addWidget(self._styled_label(
            "  分割结构量化报告（基于模型预测）",
            f"font-size:13px;font-weight:bold;color:{C['cyan']};padding:4px;"))
        te = QTextEdit(); te.setReadOnly(True); te.setFont(QFont("Courier New", 10))
        te.setText("\n".join(lines)); lay.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Ok); bb.accepted.connect(dlg.accept)
        bb.setStyleSheet(f"QPushButton{{background:{C['bg_card']};border:1px solid {C['border']};"
                         f"color:{C['cyan']};padding:4px 16px;}}")
        lay.addWidget(bb); dlg.exec_()

    def _action_settings(self):
        QMessageBox.information(self, "系统设置", "系统设置面板（预留扩展）\n当前版本 v2.4")


# ─── Entry Point ────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpineVision AI")
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(C['bg_deep']))
    palette.setColor(QPalette.WindowText,      QColor(C['text_bright']))
    palette.setColor(QPalette.Base,            QColor(C['bg_card']))
    palette.setColor(QPalette.AlternateBase,   QColor(C['bg_panel']))
    palette.setColor(QPalette.ToolTipBase,     QColor(C['bg_card']))
    palette.setColor(QPalette.ToolTipText,     QColor(C['text_bright']))
    palette.setColor(QPalette.Text,            QColor(C['text_bright']))
    palette.setColor(QPalette.Button,          QColor(C['bg_card']))
    palette.setColor(QPalette.ButtonText,      QColor(C['text_bright']))
    palette.setColor(QPalette.Highlight,       QColor(C['cyan']))
    palette.setColor(QPalette.HighlightedText, QColor(C['bg_deep']))
    app.setPalette(palette)
    win = SpineVisionApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
