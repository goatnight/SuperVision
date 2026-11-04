"""FastAPI 服务层。

API 只处理病例、文件和人工复核，不在路由中编写模型逻辑；所有推理均通过
SpineVisionWorkflow 和 Tool 调用，以便测试、追踪和以后独立部署模型服务。
"""

from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from .config import AgentSettings
from .schemas import CaseCreateRequest, CaseRecord, ReviewRequest, RunStatus
from .workflow import SpineVisionWorkflow


def create_app(workflow: SpineVisionWorkflow | None = None) -> FastAPI:
    """创建应用；测试时可注入使用 Mock Tool 的 workflow。"""

    service = workflow or SpineVisionWorkflow(AgentSettings.from_env())
    app = FastAPI(
        title="SpineVision-Agent API",
        version="0.1.0",
        description="Research-only lumbar MRI segmentation, candidate classification and grounded reporting workflow.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_origin_regex=r"https://.*\.(?:pages\.dev|workers\.dev|github\.io)$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_case(case_id: str) -> CaseRecord:
        try:
            return service.store.get(case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "SpineVision-Agent",
            "mode": "research_only",
            "segmentation_backend": "SwinUMambaLite",
            "classification_backend": service.settings.classifier_backend,
            "clinical_policy": str(service.settings.clinical_policy),
        }

    @app.get("/tools")
    def tools():
        return service.registry.describe()

    @app.post("/cases", response_model=CaseRecord)
    def create_case(request: CaseCreateRequest):
        return service.create_case(request)

    @app.get("/cases", response_model=list[CaseRecord])
    def list_cases():
        return service.store.list()

    @app.post("/cases/upload", response_model=CaseRecord)
    async def upload_case(
        file: UploadFile = File(...),
        user_goal: str = Form("完成腰椎 MRI 分割、病变候选识别及结构化报告草稿"),
        patient_hash: str | None = Form(default=None),
    ):
        # 只保留文件名，避免上传值携带目录穿越；正式部署还应增加病毒扫描和配额。
        filename = Path(file.filename or "case.nii.gz").name
        if not filename.lower().endswith((".nii", ".nii.gz")):
            raise HTTPException(status_code=422, detail="Only .nii and .nii.gz files are accepted")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")
        if len(content) > 512 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds the 512 MB research limit")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
        # 每个上传使用随机目录，避免同名文件覆盖；上传目录已在 .gitignore 中排除。
        upload_dir = service.settings.artifact_root / "uploads" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=True)
        image_path = upload_dir / safe_name
        image_path.write_bytes(content)
        request = CaseCreateRequest(
            image_path=image_path,
            modality="MRI",
            user_goal=user_goal,
            patient_hash=patient_hash,
        )
        return service.create_case(request)

    @app.get("/cases/{case_id}", response_model=CaseRecord)
    def read_case(case_id: str):
        return get_case(case_id)

    @app.get("/cases/{case_id}/status")
    def case_status(case_id: str):
        case = get_case(case_id)
        return {
            "case_id": case.case_id,
            "status": case.status,
            "current_agent": case.current_agent,
            "human_review_required": case.human_review_required,
        }

    @app.get("/cases/{case_id}/trace")
    def case_trace(case_id: str):
        return get_case(case_id).tool_trace

    @app.get("/cases/{case_id}/preview")
    def case_preview(
        case_id: str,
        slice_index: int | None = Query(default=None, ge=0),
        overlay: bool = True,
        opacity: float = Query(default=0.45, ge=0, le=1),
    ):
        case = get_case(case_id)
        try:
            import nibabel as nib

            # mmap 避免为了查看单张切片把完整 3D 体数据一次性读入内存。
            image = nib.load(str(case.image_path), mmap=True)
            if len(image.shape) != 3:
                raise ValueError("Preview requires a 3D NIfTI volume")
            index = image.shape[-1] // 2 if slice_index is None else min(slice_index, image.shape[-1] - 1)
            plane = np.asarray(image.dataobj[:, :, index], dtype=np.float32)
            finite = plane[np.isfinite(plane)]
            nonzero = finite[finite != 0]
            sample = nonzero if nonzero.size else finite
            if sample.size:
                low, high = np.percentile(sample, [1, 99])
                if high <= low:
                    high = low + 1.0
                gray = np.clip((plane - low) / (high - low), 0, 1)
            else:
                gray = np.zeros_like(plane)
            rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)

            # overlay 仅用于工作台阅片，不参与模型推理或诊断计算。
            if overlay and case.segmentation and Path(case.segmentation.mask_path).is_file():
                mask_image = nib.load(str(case.segmentation.mask_path), mmap=True)
                mask = np.asarray(mask_image.dataobj[:, :, index], dtype=np.uint8)
                palette = np.asarray(
                    [
                        [0, 0, 0], [48, 196, 141], [55, 149, 224], [234, 179, 63],
                        [221, 89, 95], [140, 108, 212], [61, 188, 196], [231, 124, 72],
                        [103, 176, 84], [202, 102, 166], [95, 130, 205], [215, 155, 52],
                        [76, 190, 151],
                    ],
                    dtype=np.uint8,
                )
                active = mask > 0
                colors = palette[np.clip(mask, 0, len(palette) - 1)]
                rgb[active] = (
                    rgb[active].astype(np.float32) * (1 - opacity)
                    + colors[active].astype(np.float32) * opacity
                ).astype(np.uint8)
            rgb = np.rot90(rgb)
            output = io.BytesIO()
            Image.fromarray(rgb).save(output, format="PNG", optimize=True)
            return Response(
                content=output.getvalue(),
                media_type="image/png",
                headers={
                    "X-Slice-Index": str(index),
                    "X-Slice-Count": str(image.shape[-1]),
                    "Cache-Control": "no-store",
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Unable to render preview: {exc}") from exc

    @app.post("/cases/{case_id}/run", response_model=CaseRecord)
    def run_case(case_id: str):
        get_case(case_id)
        try:
            return service.run_case(case_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/cases/{case_id}/review", response_model=CaseRecord)
    def review_case(case_id: str, request: ReviewRequest):
        # 人工反馈被保存在病例状态中，后续可整理成 Report SFT/偏好学习数据。
        case = get_case(case_id)
        case.human_feedback.append(request.model_dump(mode="json"))
        if request.decision == "accept":
            case.status = RunStatus.completed
            case.human_review_required = False
            if case.report:
                case.report.status = "reviewed"
        elif request.decision == "modify":
            if request.corrected_report is None:
                raise HTTPException(status_code=422, detail="corrected_report is required for modify")
            case.report = request.corrected_report
            case.report.status = "reviewed"
            case.status = RunStatus.completed
            case.human_review_required = False
        else:
            case.status = RunStatus.failed
            case.human_review_required = True
        return service.store.save(case)

    app.state.workflow = service
    return app


app = create_app()
