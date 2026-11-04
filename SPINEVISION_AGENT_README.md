# SpineVision-Agent MVP

本模块在不修改现有训练代码的前提下，将腰椎 MRI 分割、ROI 分类、知识检索、报告生成和一致性校验封装为 LangGraph 工作流与 FastAPI 服务。

## 当前能力

- `Planner`：使用受限 ReAct 计划，只能调用注册表中的工具；
- `Segmentation`：支持加载本地 SwinUMambaLite checkpoint，逐矢状位切片推理并保留 NIfTI affine/header；
- `Diagnosis`：支持 ROI 辅助病例级滑脱分类 checkpoint、温度校准和双阈值拒识；公开版本默认使用 mock；
- `RAG`：使用 SQLite 持久化向量库和离线 Hashing embedding，返回可追溯 `evidence_id`；
- `Report`：只根据分割、分类和检索证据生成结构化草稿；
- `Verifier`：检查分割 QC、分类置信度、具体节段和证据引用；
- `Human Review`：低置信度、高风险冲突或标签语义未验证时强制人工复核。

系统仅用于研究和工程验证，不输出未经医生复核的最终诊断。

## 安装与启动

```powershell
& 'D:\Application\CodeEditor\anaconda3\envs\pytorch\python.exe' -m pip install -r requirements-agent.txt
& 'D:\Application\CodeEditor\anaconda3\envs\pytorch\python.exe' run_agent_api.py
```

访问：

- OpenAPI：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`
- 工具清单：`http://127.0.0.1:8000/tools`

## API 流程

创建病例：

```json
POST /cases
{
  "image_path": "H:/data/Case001.nii.gz",
  "modality": "MRI",
  "user_goal": "完成腰椎 MRI 分割、病变候选识别及结构化报告草稿"
}
```

依次调用：

```text
POST /cases/{case_id}/run
GET  /cases/{case_id}/status
GET  /cases/{case_id}/trace
POST /cases/{case_id}/review
```

## 真实模型配置

公开仓库不包含医学模型权重。接入本地真实模型时使用 `roi_case` 后端：

```text
SPINEVISION_SEGMENTATION_PROJECT=train_test
SPINEVISION_SEGMENTATION_CHECKPOINT=runslite_256/best.pt
SPINEVISION_CLASSIFIER_BACKEND=roi_case
SPINEVISION_CLASSIFIER_PROJECT_ROOT=remote_training/spine_roi_case_cls
SPINEVISION_CLASSIFIER_CHECKPOINT=remote_training/spine_roi_case_cls/runs/best.pt
SPINEVISION_CLINICAL_POLICY=remote_training/spine_roi_case_cls/runs/clinical_policy.json
```

真实链路执行：训练域输入 QC → `SwinUMambaLite` 全卷分割 → 中央 3 层图像/ROI 双分支分类 → 温度校准 → 双阈值拒识。分类 checkpoint 与分割 checkpoint 必须同时匹配 `clinical_policy.json` 中的 SHA-256，否则 Tool fail closed。

服务器路径示例见 `.env.agent.remote.example`。未配置私有权重时，默认 `mock` 后端仅用于工程演示和单元测试。

## 真实集成验收

服务器使用一例保留测试病例实测通过两条入口：

```bash
PYTHONPATH=. python tests/run_real_model_smoke.py --image CASE.nii.gz --output real_case.json
PYTHONPATH=. python tests/run_real_api_smoke.py --image CASE.nii.gz --output real_api_case.json
```

实测 trace 为 `inspect → segmentation → diagnosis → RAG → report → verifier` 共 6 步；分割检出 1-12 类，真实分类器输出原始概率 `0.3101`、校准概率 `0.3151`。由于概率位于拒识区间且策略未获临床授权，最终状态为 `review`，没有自动完成诊断。

## 重要限制

1. 当前 checkpoint 的 12 个前景标签正式语义尚未由标注字典确认，因此 Verifier 会强制进入人工复核。
2. 当前 RAG 种子仅包含内部安全 SOP，不是临床指南库；导入正式指南前不得声称提供临床证据。
3. 分类后端已经真实接入，但资格策略为 `NOT_CLINICALLY_QUALIFIED`；所有结果仍强制医生复核。
4. `InMemoryCaseStore` 适合单进程 MVP；生产环境应替换为 Redis/PostgreSQL 并增加访问控制和 DICOM 去标识化。
