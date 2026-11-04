# SpineVision-Agent 项目执行与监管提示词

> 用途：本文件既是 AI 编码助手的项目提示词，也是项目实施期间的进度、验收、风险和决策记录。
>
> 使用方式：开始新任务时，将“主执行提示词”和当前阶段内容交给 Codex/Claude Code；完成工作后，必须同步更新阶段状态、验收证据、风险与进度日志。

## 文档状态

| 字段 | 当前值 |
|---|---|
| 项目名称 | SpineVision-Agent |
| 当前阶段 | Phase 1：真实分割/分类模型接入完成，待项目负责人验收 |
| 项目状态 | 待验收 |
| 项目负责人 | 待填写 |
| 最近更新日期 | 2026-08-07 |
| 当前阻塞项 | `1..12` 正式标签语义、正式医学指南知识库、外部验证和 LLM 节点尚未完成 |
| 下一里程碑 | 接入 Qwen Planner/Report/Verifier，并保持真实模型 Tool 为确定性事实源 |

状态只能使用：`未开始`、`进行中`、`受阻`、`待验收`、`已完成`。

---

## 一、主执行提示词

你是一名资深医疗影像 AI 架构师、LLM/VLM Agent 工程师和 Python 技术负责人。你需要基于当前工作区中的已有脊柱/腰椎医学影像分割项目，设计并逐步实现一个可运行、可演示、可评测的医疗影像 Agent 系统。

项目名称：

**SpineVision-Agent：基于腰椎分割模型的多 Agent 影像分析、定量测量、报告生成与质量控制系统**

这不是从零重写分割模型，也不是只输出概念方案。必须先阅读现有仓库，识别数据流、模型入口和推理逻辑，再以最小侵入方式封装已有能力，逐阶段交付可运行代码。

### 1. 项目目标

建立以下完整工作流：

```text
医学影像输入
→ 数据完整性与序列检查
→ 分割模型调用
→ 分割结果后处理与质量控制
→ 解剖结构识别及椎体节段定位
→ 定量测量
→ 异常候选发现
→ 医学知识检索
→ 结构化影像所见及报告草稿生成
→ 图像、测量、描述一致性验证
→ 不确定性和风险判断
→ 人工复核
→ 审计日志与反馈保存
```

系统定位是医学影像分析辅助与研究原型，不是独立诊断系统。诊断性内容必须标记为候选结果或报告草稿，并保留医生复核环节。

项目需重点体现：

- LLM/VLM 应用、Planner、Tool-Use 与结构化输出；
- LangGraph 状态机、ReAct 工具调用和 Reflexion 失败恢复；
- 多 Agent 分工、协作、质控和冲突消解；
- Python 工程、模型服务封装、异步任务和错误处理；
- 医学知识 RAG、证据引用和报告 grounding；
- 将临床工作流拆成稳定、可测试的工具；
- 长程任务和 Tool-Use Benchmark；
- SFT、偏好优化、过程奖励模型和 Agentic RL；
- 医疗安全、不确定性管理、人工复核和可追溯性。

### 2. 强制原则

1. 先分析仓库，再设计改造，再实施。
2. 优先复用已有模型、配置、权重加载、预处理和推理代码。
3. 不得为了接入 Agent 重写整个分割模型。
4. 数据拆分必须以患者为单位；同一患者的切片、序列和时间点不得跨集合。
5. 测试集不得用于 Prompt、阈值、后处理参数或超参数调优。
6. 不得虚构数据规模、模型性能、临床结论或实验结果。
7. 无法确认的信息输出 `unknown` 或 `null`。
8. LLM 不计算关键医学测量值；测量由确定性 Python 工具完成。
9. LLM 不直接修改原始 mask，只能触发后处理工具或提出人工修正建议。
10. 每个结论必须追溯到图像、mask、测量、规则或检索证据。
11. 高风险、低置信度、工具失败或结果冲突时必须转人工复核。
12. 不保存隐藏思维链；只保存结构化决策摘要、工具轨迹、证据和错误原因。
13. 未读取代码前不得猜测文件名、模型接口、类别定义和数据格式。
14. 每个阶段必须先通过验收，再进入下一阶段。

### 3. 现有仓库审计要求

Phase 0 只进行只读分析。使用 `rg`、`rg --files` 等工具检查：

- Python、PyTorch、MONAI、nnU-Net 等技术栈；
- 模型定义文件、模型类和初始化参数；
- 训练、验证和推理入口；
- checkpoint 加载方式及路径配置；
- 输入格式、维度、模态和序列要求；
- 重采样、spacing、orientation、裁剪、归一化和 resize；
- mask shape、类别映射、概率图和后处理；
- 可视化、导出、配置、日志、测试和依赖；
- 可直接复用、需要适配和需要新增的组件；
- 患者泄漏、几何错位、spacing 丢失、路径硬编码和 GPU 耦合等风险。

必须输出“现有模块 → Agent 工具”映射表：

| 现有模块 | 文件路径 | 当前功能 | 可封装工具 | 所需改造 | 风险等级 |
|---|---|---|---|---|---|
| 待审计 | 待确认 | 待确认 | 待确认 | 待确认 | 待确认 |

除非现有推理完全不可用，否则不重新训练模型。

### 4. 临床使用场景

根据仓库真实能力限定场景，不得超过现有标签和模型能力。优先考虑：

- 腰椎 MRI/CT 读取和完整性检查；
- 椎体、椎间盘、椎管或病灶结构分割；
- 腰椎节段定位；
- mask 体积、面积、长度、角度或距离测量；
- 左右侧、节段级和结构级结果汇总；
- 异常候选区域提示；
- 结构化影像所见和报告草稿；
- 分割、测量、文本之间的一致性检查；
- 失败分割、低质量图像和异常测量的人工复核。

必须明确：目标人群、模态、使用者、预期用途、主要任务、主要终点、参考标准、分析单位、排除条件、人工复核点、系统不做什么、MVP 流程和增强版流程。

### 5. 系统架构

采用三层架构：

- **确定性医学工具层**：图像处理、模型推理、测量、规则计算和结果验证；
- **Agent 编排层**：使用 LangGraph 管理显式状态、条件分支、重试和终止；
- **人机交互层**：FastAPI 提供接口，Streamlit 或现有前端提供病例工作台。

核心角色：

| Agent/节点 | 职责 |
|---|---|
| Case Intake | 检查文件、模态、序列、元数据和模型适配性 |
| Planner | 生成结构化计划，只能选择已注册工具 |
| Segmentation | 调用已有模型并记录版本、参数和耗时 |
| Measurement | 调用确定性工具计算物理测量值 |
| Knowledge/RAG | 检索指南、标准、术语和报告模板 |
| Report | 基于结构化发现和证据生成报告草稿 |
| Verifier/Critic | 检查数字、左右侧、节段、证据和文本一致性 |
| Safety Gate | 判断继续、重试、终止或转人工 |
| Human Review | 接受、修改、驳回并记录理由 |

MVP 先使用单图状态机和专业节点；完成基线后再拆成多 Agent，避免无效复杂度。

### 6. LangGraph 状态与流程

`CaseState` 至少包含：

```text
case_id, patient_hash, study_id, modality, series_metadata,
input_files, image_manifest, preprocessing_record,
model_version, checkpoint_hash, segmentation_outputs,
segmentation_qc, anatomy_levels, measurements,
candidate_findings, retrieved_evidence, report_draft,
verification_results, uncertainty, execution_plan,
completed_steps, tool_trace, retry_count, error_records,
human_review_required, human_feedback, final_status, timestamps
```

全部字段使用 Pydantic 定义，不在节点间传递含义不明的自由文本。

主流程：

```text
START → validate_input → build_plan → preprocess
→ run_segmentation → segmentation_qc → identify_anatomy
→ calculate_measurements → derive_candidate_findings
→ retrieve_knowledge → generate_report → verify_case
→ risk_gate → human_review/finalize → END
```

必须覆盖：输入缺失、不支持模态、模型加载失败、GPU OOM、推理超时、空 mask、异常连通域、mask 越界、spacing 缺失、测量异常、报告冲突、RAG 无证据和重试失败。

每个节点定义输入、输出、前置条件、成功条件、失败类型、重试次数、超时、下一节点和审计信息。

### 7. 工具协议

每个工具必须声明：

```text
name, description, version, input_schema, output_schema,
preconditions, deterministic, timeout_seconds, retry_policy,
error_codes, provenance, requires_human_review
```

统一返回：

```json
{
  "status": "success | partial | failed | rejected",
  "data": {},
  "warnings": [],
  "errors": [],
  "metrics": {},
  "provenance": {
    "tool_name": "",
    "tool_version": "",
    "model_version": null,
    "checkpoint_hash": null,
    "input_hash": "",
    "started_at": "",
    "finished_at": ""
  }
}
```

至少实现或预留：

```text
inspect_input_files, load_dicom_series, load_nifti_volume,
validate_series_metadata, convert_dicom_to_nifti, preprocess_volume,
run_spine_segmentation, restore_mask_to_original_space,
postprocess_mask, segmentation_quality_control,
identify_connected_components, identify_vertebral_levels,
calculate_mask_volume, calculate_cross_sectional_area,
calculate_distance, calculate_angle, calculate_structure_statistics,
derive_candidate_findings, retrieve_guideline_evidence,
retrieve_report_template, generate_structured_findings,
generate_report_draft, verify_numeric_consistency,
verify_anatomical_consistency, verify_report_grounding,
create_overlay_preview, export_case_result, request_human_review
```

每个工具都要有工具卡、JSON 示例、错误码、单元测试和转人工条件。

### 8. 分割模型适配与质量控制

将现有模型封装为 `SegmentationService`，保持模型核心结构不变：

```python
class SegmentationService:
    def load_model(self) -> None: ...
    def validate_input(self, image): ...
    def preprocess(self, image): ...
    def predict(self, tensor): ...
    def postprocess(self, prediction): ...
    def restore_geometry(self, mask): ...
    def quality_control(self, image, mask): ...
```

记录模型配置、checkpoint hash、device、shape、spacing、orientation、归一化、推理耗时、阈值、后处理参数和类别。支持模型缓存、CPU/GPU 选择、显存不足回退、无梯度推理和原始物理空间恢复。

QC 至少检查：空 mask、越界、异常连通域、异常体积、解剖位置、层间突变、孔洞碎片、类别重叠、几何不一致和低置信度。无真实数据依据的阈值标记为“待校准”。

### 9. 测量与候选发现

所有测量使用物理坐标和真实 spacing，禁止将像素数直接当作毫米或体积。

结果包含：测量名称、节段、侧别、数值、单位、方法、来源 mask、来源序列、质量标记、不确定性和 provenance。

发现状态必须区分：

- `observed`：工具直接观察或测量；
- `derived`：规则计算；
- `inferred`：模型或 LLM 推断；
- `unknown`：证据不足。

LLM 只能组织和解释 `observed`、`derived`，不得把 `inferred` 改写成确定性诊断。

### 10. RAG

知识库包含结构化报告模板、解剖术语、分级标准、公开指南、测量定义和项目 SOP。条目保留文档 ID、标题、机构、年份、版本、章节、模态、解剖部位、任务类型、来源和 hash。

采用语义分块、章节保留、BM25 + vector 混合检索、元数据过滤和可选 reranker。返回证据 ID 和来源；无可靠证据时返回 `insufficient_evidence`，不得生成伪引用。

### 11. Prompt 与结构化输出

分别实现 Planner、Tool Selection、Report、Verifier、Reflection 和 Human Review Summary Prompt。所有关键输出使用 Pydantic/JSON Schema 校验。

Planner 输出：目标、已知事实、缺失信息、步骤、工具参数、成功条件、失败动作、停止条件和转人工条件。

Report 输出：检查信息、技术、图像质量、分节段 findings、证据 ID、置信度、印象、限制和复核建议。

Verifier 输出：是否通过、错误类型、严重程度、证据、缺失项、冲突、建议动作和是否转人工。

Prompt 必须防止编造测量、虚构节段、左右混淆、将失败工具视为成功、无依据诊断、隐藏不确定性和伪造文献。

### 12. Reflexion 与多 Agent

仅在参数错误、可恢复工具失败、Verifier 冲突、计划缺步、schema 不合法或检索不足时触发 Reflection。每步最多重试 1～2 次；相同参数不得重复调用失败工具；critical 错误直接转人工。

多 Agent 阶段包含 Planner、Executor、Radiology、Knowledge、Critic 和 Safety。确定性工具结果优先于 LLM 文本，几何和测量优先于语言推断；无法消解的高风险冲突转人工。禁止无限协商循环。

### 13. SpineAgent-Bench

任务类型：

- T1 单工具选择；
- T2 工具参数生成；
- T3 多步流程规划；
- T4 故障恢复；
- T5 跨工具一致性；
- T6 安全拒答和转人工；
- T7 受预算约束的长程任务；
- T8 多 Agent 协作和冲突终止。

每个样本包含用户目标、初始状态、可用工具、标准计划、标准工具序列、参数、中间状态、最终状态、注入故障和安全预期。

评估指标包括：工具选择准确率、参数准确率、计划有效率、步骤完成率、端到端成功率、非法/冗余调用率、恢复成功率、报告 grounding、数字一致性、幻觉率、转人工 precision/recall、调用次数、延迟和 token 成本。

### 14. Agent 后训练与过程奖励

实施顺序：

1. 采集脱敏的成功/失败工具轨迹；
2. 构建 Planner、Tool Calling、Verifier、Reflection 的 SFT 数据；
3. 构建 chosen/rejected 轨迹对进行 DPO/ORPO；
4. 训练或实现 PRM，对每一步进行过程评分；
5. 离线稳定后再研究 PPO/GRPO，禁止在真实患者流程中无控制探索。

过程奖励参考：

```text
r_t =
  w1 * tool_selection_correct
+ w2 * argument_valid
+ w3 * state_progress
+ w4 * evidence_grounded
+ w5 * consistency
+ w6 * uncertainty_awareness
- w7 * invalid_call
- w8 * redundant_call
- w9 * hallucination
- w10 * unsafe_completion
```

必须输出 trajectory schema、SFT 格式、偏好对格式、PRM 标签规范、奖励函数、离线评测、防 reward hacking 和消融实验。

### 15. 验证要求

分别验证：分割模型、测量工具、Agent 工具使用、报告一致性和临床用户工作流。

分割指标可包含 Dice、IoU、HD95、ASD、敏感度、分类别结果和失败率；测量验证可包含 ICC、Bland-Altman 和绝对误差；Agent 使用 SpineAgent-Bench。

完整 pipeline 冻结后再进行患者级测试和外部/时间外验证。若当前数据不支持外部验证，必须明确写成限制和未来计划。

### 16. 工程要求

根据现有仓库调整目录，不机械创建重复模块。目标模块包括：`api`、`agents`、`graph`、`tools`、`model_adapter`、`rag`、`benchmark`、`training`、`observability`、`ui`、`configs` 和 `tests`。

必须具备 Python 类型标注、Pydantic、配置分离、结构化日志、统一异常、工具注册、依赖注入、mock 模型、mock LLM、单元测试、集成测试、环境变量示例和 README。不得提交真实患者数据或密钥。

FastAPI 至少提供病例创建、运行、状态、轨迹、复核、健康检查和工具列表接口。演示界面必须是病例工作台，展示原图、mask overlay、QC、测量、证据、报告、冲突和人工复核，不得只有聊天框。

### 17. 可观测性

每次运行记录 case ID、graph run ID、Prompt 版本、LLM 模型、工具和版本、输入输出 hash、checkpoint hash、延迟、token、重试、错误、证据、人工反馈和最终状态。

建立以下追踪链：

```text
报告句子 → finding_id → measurement_id/mask_id
→ tool call → model/checkpoint → 输入影像 hash
```

普通日志不得保存患者姓名、证件号、敏感 DICOM 标签或未脱敏 Prompt。

### 18. 执行纪律

每阶段开始前：

1. 阅读本文件；
2. 检查当前状态、阻塞项和上一阶段证据；
3. 输出本轮计划和预计修改文件；
4. 确认不会破坏已有分割流程。

每阶段结束时：

1. 运行对应测试；
2. 更新本文件的阶段状态；
3. 填写验收证据和已知限制；
4. 更新风险登记和决策记录；
5. 将未验证内容明确标记为待验证；
6. 不得仅凭“代码已写完”将阶段标记为完成。

### 19. 当前执行命令

现在从 Phase 0 开始。只读分析当前仓库，输出：

1. 技术栈和目录结构；
2. 模型定义和推理入口；
3. 数据读取及预处理链；
4. checkpoint 加载方式；
5. 类别定义和 mask 格式；
6. 可复用模块；
7. 风险及缺失信息；
8. 现有模块到 Agent 工具的映射表；
9. 最小侵入改造方案；
10. Phase 1 计划及验收条件。

遇到未知信息时标记“待确认”。完成 Phase 0 后暂停大规模改造，先提交审计结果和本文件的状态更新。

---

## 二、阶段进度看板

| 阶段 | 目标 | 状态 | 负责人 | 开始日期 | 完成日期 | 验收证据 |
|---|---|---|---|---|---|---|
| Phase 0 | 仓库审计与最小改造方案 | 待验收 | Codex（审计） | 2026-07-31 | 2026-07-31 | `PHASE0_REPOSITORY_AUDIT.md`；15/15 AST 解析通过；2 个 checkpoint 合成前向通过；患者泄漏复现实验 |
| Phase 1 | 分割与分类模型工具化、输入/ROI QC | 待验收 | Codex | 2026-08-07 | 2026-08-10 | 真实 MRI 六步 LangGraph 与 FastAPI 冒烟通过；双 checkpoint/策略 hash 可追溯 |
| Phase 2 | 确定性测量与后处理工具 | 前置实现（未验收） | Codex | 2026-07-31 | - | `spinevision_clinical_metrics.py`；12/12 单元/契约测试通过；尚缺真实病例验证 |
| Phase 3 | LangGraph MVP | 待验收 | Codex | 2026-08-07 | - | 6 步 LangGraph 工具链测试通过；状态、异常和 trace 可追踪 |
| Phase 4 | RAG 与结构化报告 | 待验收 | Codex | 2026-08-07 | - | SQLite 向量检索、evidence_id 注入和结构化报告测试通过；当前仅内部 SOP |
| Phase 5 | Verifier 与人工复核 | 待验收 | Codex | 2026-08-07 | - | 标签语义、低置信度、证据引用校验与 review API 已实现 |
| Phase 6 | SpineAgent-Bench | 待验收 | Codex | 2026-08-07 | - | `SPINEAGENT_BENCHMARK_REPORT.md`；30 例编排、5 例真实分割、8 个 RAG 查询、120 个不一致、50 个故障注入 |
| Phase 7 | 多 Agent 与 Reflexion | 未开始 | 待填写 | - | - | - |
| Phase 8 | SFT、偏好数据与 PRM | 未开始 | 待填写 | - | - | - |

## 三、阶段验收清单

### Phase 0：仓库审计

- [x] 已识别模型定义、训练入口和推理入口
- [ ] 已确认输入格式、维度、模态和序列要求（训练文件格式为 2D 矢状位 NIfTI；DICOM/PNG 保留兼容；MRI/CT 及具体序列待确认）
- [x] 已梳理完整预处理和后处理链
- [ ] 已确认 checkpoint 加载方式和类别映射（已确认背景 + 12 前景、`best.pt` 基线并修复 GUI 数量；`1..12` 正式名称/顺序待确认）
- [x] 已检查患者级拆分及潜在泄漏（已证实默认切片级拆分存在患者泄漏）
- [x] 已检查 spacing、affine、orientation 和 mask 几何完整性（已识别缺失校验及 GUI 导出 affine 丢失）
- [x] 已完成“现有模块 → Agent 工具”映射
- [x] 已形成最小侵入改造方案
- [x] 未猜测无法从仓库确认的信息

### Phase 1：模型工具化

- [x] `SegmentationService` 可独立加载现有模型
- [ ] 现有推理结果与适配器结果一致
- [x] 输入输出均通过 Pydantic/schema 校验
- [x] 保存模型版本、checkpoint hash 和预处理记录
- [ ] 支持 CPU/GPU 与显存不足回退
- [x] mask 可恢复到原始物理空间
- [x] 基础 QC 能识别空 mask并保持输入输出几何一致
- [x] 已通过单元测试和至少一个脱敏/合成病例测试
- [x] 病例级 ROI 分类 checkpoint 已作为默认后端接入
- [x] 输入 QC、ROI QC、校准概率、拒识状态和模型指纹进入病例状态

### Phase 2：确定性工具

- [ ] 已实现后处理和连通域分析
- [x] 至少实现一种有临床意义的物理测量（条件性 L1–S1 上终板腰椎前凸角）
- [x] 测量使用真实 spacing 和物理单位（NIfTI affine / DICOM 几何；mask 面积/体积）
- [ ] 工具包含错误码、provenance 和测试
- [x] 阈值集中配置，未经验证的阈值标记为待校准（本次已删除所有未验证诊断阈值；未引入替代阈值）

### Phase 3：LangGraph MVP

- [x] 已实现显式 `CaseState`
- [x] 已串联输入、分割、诊断、RAG、报告、校验和结束节点
- [x] 已覆盖工具失败、低置信度和人工复核分支
- [x] 已限制重试次数，无法无限循环
- [x] 使用 mock 分类后端时可离线完成测试
- [x] 工具轨迹和状态变化可审计

### Phase 4：RAG 与报告

- [x] 知识来源合法且保留版本元数据
- [ ] 支持混合检索和元数据过滤
- [x] 报告内容能追溯到 finding 和 evidence ID
- [x] 检索不到证据时返回 `insufficient_evidence`
- [x] 报告结构通过 schema 校验
- [x] 未将推断内容改写成确定性诊断

### Phase 5：Verifier 与人工复核

- [ ] 可检查数字、单位、节段和左右侧
- [x] 可检查报告、分割 QC、分类置信度和证据引用一致性
- [x] 可识别低质量、冲突和高风险结果
- [x] 人工可接受、修改或驳回
- [x] 人工修改及理由进入审计记录

### Phase 6：Benchmark

- [x] 至少 30 个工具级测试场景
- [x] 至少 10 个端到端场景
- [x] 覆盖故障注入、安全拒答和转人工
- [x] 支持规则自动评分
- [ ] 输出准确率、完成率、幻觉、成本和延迟
- [x] Benchmark 测试数据与开发调试数据隔离

### Phase 7：多 Agent 与 Reflexion

- [ ] Agent 间消息使用结构化协议
- [ ] 确定性结果优先级明确
- [ ] 冲突可被 Verifier 或人工终止
- [ ] Reflection 仅在允许条件下触发
- [ ] 已对比单 Agent 与多 Agent 的收益和成本

### Phase 8：后训练与 PRM

- [ ] 轨迹已脱敏并通过 schema 校验
- [ ] SFT 数据覆盖工具选择、参数、验证和恢复
- [ ] 偏好对有明确 chosen/rejected 原因
- [ ] PRM 能输出步骤级奖励
- [ ] 已检查 reward hacking
- [ ] 已完成离线评估，未在真实患者流程在线探索

## 四、MVP 总验收标准

- [ ] 能调用已有分割模型而非替代它
- [ ] 能输出结构化 mask 元数据和 QC
- [ ] 能完成至少一种真实物理量测量
- [ ] 能生成带证据的结构化报告草稿
- [ ] 能验证报告与工具结果一致性
- [ ] 失败、冲突和低置信度结果能停止或转人工
- [ ] 能展示完整工具调用轨迹
- [ ] 能运行自动化 Benchmark
- [ ] 无真实 LLM API 时可使用 mock 模式测试
- [ ] README 包含安装、配置、运行、测试和限制
- [ ] 不含真实患者隐私、密钥和未经授权的数据

## 五、风险登记表

| ID | 风险 | 严重度 | 触发/检测方式 | 缓解措施 | 状态 | 负责人 |
|---|---|---|---|---|---|---|
| R-01 | 输入序列不符合模型要求 | 高 | 元数据及 shape 校验 | 输入门控；兼容但未验证的输入显式警告 | 已识别：训练使用 NIfTI，具体模态/序列待确认；DICOM/PNG 保留兼容 | 待填写 |
| R-02 | 分割模型存在域偏移 | 高 | 外部/时间外失败率 | 不确定性、QC、人工复核 | 未评估 | 待填写 |
| R-03 | mask 与原图几何错位 | 严重 | affine/spacing/shape 检查 | 恢复原空间并强制验证 | 已证实：GUI NIfTI 导出使用单位 affine；训练未校验图像/标签几何 | 待填写 |
| R-04 | 节段或左右侧识别错误 | 严重 | 解剖一致性规则 | Verifier + 人工复核 | 数量冲突已修复为 12 前景；正式名称/顺序仍待确认 | 待填写 |
| R-05 | 测量单位错误 | 严重 | 物理坐标和单位校验 | 禁止像素值直接作为测量 | 已识别：当前 spacing 为字符串解析，未做 affine/单位验证 | 待填写 |
| R-06 | LLM/规则编造或过度解释影像事实 | 严重 | grounding 检查 | 仅使用结构化 finding 与证据 | 已缓解：删除未校准狭窄/灰度骨密度/椎间隙/Cobb-like 结论；条件不足时 fail-closed，仍需真实病例验证 | 待填写 |
| R-07 | RAG 引用错误或过期 | 高 | 来源、版本和日期校验 | 版本化知识库和无证据回退 | 未评估 | 待填写 |
| R-08 | Prompt injection/非法工具调用 | 高 | 工具白名单和 schema | 参数校验、权限边界 | 未评估 | 待填写 |
| R-09 | 患者隐私泄露 | 严重 | 日志和导出审计 | 去标识化、hash 和最小日志 | 未评估 | 待填写 |
| R-10 | 草稿被误作最终诊断 | 严重 | UI/导出状态检查 | 明确草稿水印和强制人工确认 | 未评估 | 待填写 |
| R-11 | Agent 无限循环或成本失控 | 中 | 调用和 token 预算 | 最大步数、重试和超时 | 未评估 | 待填写 |
| R-12 | Benchmark 与真实任务不一致 | 中 | 错误分布比较 | 增加真实失败模式与外部测试 | 未评估 | 待填写 |
| R-13 | 默认切片级拆分导致患者泄漏 | 严重 | 比较 train/validation 的 `pa_id` | 改为患者级分组拆分并增加零交集测试；旧指标降级说明 | 已证实 | 待填写 |
| R-14 | checkpoint 与 GUI 标签数量/语义冲突 | 严重 | 比较输出通道、背景约定和 GUI label 表 | 已修复为背景 + 12 前景并添加契约测试；继续确认正式名称/顺序 | 数量已缓解，语义待确认 | 待填写 |
| R-15 | 逐切片 Dice 和空类计分造成指标乐观偏倚 | 高 | 审计聚合单位与空类策略 | 患者/体积级重评估、置信区间与失败率 | 已证实 | 待填写 |
| R-16 | `best.pt` 与 `last.pt` 并非同一训练运行 | 中 | 比较 checkpoint config 与模型宽度 | 运行 ID/manifest 管理；Phase 1 已选 `best.pt` | 基线已确认，运行溯源仍待完善 | 待填写 |

## 六、关键决策记录

| 日期 | 决策 ID | 决策 | 依据 | 影响 | 决策人 |
|---|---|---|---|---|---|
| 2026-07-16 | ADR-001 | 复用现有分割模型，以工具适配器接入 Agent | 降低改造风险并保留已有能力 | Phase 0 需先定位真实推理链 | 待确认 |
| 2026-07-16 | ADR-002 | MVP 使用 LangGraph 单图多节点 | 先验证工具和状态管理价值 | 多 Agent 延后到 Phase 7 | 待确认 |
| 2026-07-16 | ADR-003 | 医学测量由确定性工具完成 | 保证单位、复现性与审计能力 | LLM 仅组织和解释结果 | 待确认 |
| 2026-07-31 | ADR-004 | Phase 1 冻结现有模型核心，以独立 `SegmentationService` 和 Pydantic 工具协议适配 | 两个现有 checkpoint 已可用；重写网络没有必要 | 保留回归基线，优先修复输入、几何、标签和审计边界 | Codex（建议，待负责人确认） |
| 2026-07-31 | ADR-005 | 在标签字典与临床阈值确认前，禁用节段诊断、骨密度、狭窄和 Cobb-like 结论 | 当前通道/标签冲突且规则未经校准 | Phase 1 仅交付分割、基础 QC 和结构化 provenance | Codex（建议，待负责人确认） |
| 2026-07-31 | ADR-006 | 模型类别契约固定为背景 `0` + 12 个前景 mask，GUI 删除第 13 个“硬膜囊”展示项 | 项目负责人确认实际输出；checkpoint 为 13 通道 | 展示标签范围固定为 `0..12`，新增自动化契约测试 | 项目负责人 |
| 2026-07-31 | ADR-007 | 训练输入文件格式记录为 NIfTI，继续兼容 DICOM/PNG，但后两者标记为待验证 | 项目负责人确认训练格式并要求保留兼容 | UI 显示输入域状态；后续分别验证预处理和性能 | 项目负责人 |
| 2026-07-31 | ADR-008 | Phase 1 选用 `runslite_256/best.pt`（640/base16）作为基线 | 项目负责人选择 | 默认模型发现路径优先使用 best；last 仅保留为历史 checkpoint | 项目负责人 |
| 2026-07-31 | ADR-009 | 临床指标采用“证据定义 + 输入门控 + 条件不足拒绝计算”；当前条件性输出 L1–S1、L4–S1 前凸角、S1 骶骨倾斜角和描述性 mask 几何量 | 现有 12 类 mask 不支持椎间盘、硬膜囊/神经根形态或 CT HU；临床标准不允许用聚合面积比/显示灰度替代 | GUI 改为“标准化测量与可用性”；不再自动生成狭窄、骨密度、退变结论或治疗建议 | 项目负责人要求，Codex 实施 |

## 七、进度日志

每次工作结束后追加一行，不覆盖历史记录。

| 日期 | 阶段 | 完成内容 | 修改文件 | 测试/证据 | 遗留问题 | 下一步 |
|---|---|---|---|---|---|---|
| 2026-07-16 | 准备 | 创建项目执行与监管文档 | `SPINEVISION_AGENT_PROJECT.md` | 文档已生成 | 尚未进行仓库审计 | 执行 Phase 0 |
| 2026-07-31 | Phase 0 | 完成技术栈、模型/推理、数据/几何、checkpoint、类别、风险和工具映射审计 | `PHASE0_REPOSITORY_AUDIT.md`、`SPINEVISION_AGENT_PROJECT.md` | 15/15 Python AST 通过；两个 checkpoint 合成 CPU 前向通过；合成拆分证实患者泄漏 | 标签字典、模态/序列、患者清单、基线 checkpoint 待确认 | 负责人验收 Phase 0 后启动 Phase 1 |
| 2026-08-07 | Agent MVP | 实现 LangGraph+ReAct 工作流、分割/分类 Tool、SQLite RAG、Report、Verifier、人工复核与 FastAPI | `spinevision_agent/`、`run_agent_api.py`、`requirements-agent.txt`、`SPINEVISION_AGENT_README.md` | 14/14 单元测试通过；真实 `best.pt` 单切片 NIfTI 冒烟通过；FastAPI health/tools 通过 | 分类仍为 mock 默认后端；正式标签语义和医学指南未接入 | 接入 `cls_unified` callable 并运行脱敏病例端到端验证 |
| 2026-08-07 | Phase 6 | 建立工程 Benchmark，实测 Tool 调用、耗时、RAG、报告契约、Verifier、异常恢复及人工步骤 | `spinevision_agent/benchmark/`、`run_agent_benchmark.py`、`SPINEAGENT_BENCHMARK_REPORT.md` | 真实分割链路 5 例；RAG 8 查询；报告 100 例；Verifier 120 错误+20 干净；恢复 50 例 | 分类为 mock；RAG 仅 4 篇内部 SOP；人工步骤非真实用户研究 | 接入真实分类、正式知识库和真实脱敏 MRI 后重跑 |
| 2026-07-31 | Phase 0 修订 | 按负责人确认修正为背景+12前景；best 为默认基线；标记 NIfTI 训练格式及 DICOM/PNG 兼容状态 | `SpineVision_AI_real_infer.py`、`tests/test_segmentation_contract.py`、相关说明与监管文档 | 16/16 AST 通过；4/4 类别契约测试通过 | 正式标签名称/顺序、底层模态/序列、患者清单仍待确认 | 验收 Phase 0 后进入 Phase 1 服务化 |
| 2026-07-31 | Phase 2 前置修订 | 按临床文献重构指标：实现 L1–S1、L4–S1 前凸角、S1 骶骨倾斜角及描述性 mask 面积/体积；JPG/PNG 在方形像素假设下开放旋转不变的两终板相对夹角；删除未校准阈值与治疗建议 | `spinevision_clinical_metrics.py`、`SpineVision_AI_real_infer.py`、`CLINICAL_METRICS_STANDARD.md`、`tests/test_clinical_metrics.py` | GUI Python 环境 12/12 测试通过；旧 2.5%/7%、55%/80%、0.75 灰度阈值已清除 | JPG/PNG 无 spacing 与患者水平轴；真实病例终板拟合精度、输入模态/序列、CT HU 路径仍未验证 | 以人工标志点为参考验证三个角度；为 raster 增加显式方向/像素间距输入 |
| 2026-08-10 | Phase 1 真实接入 | 将分割 checkpoint、ROI 病例级分类 checkpoint、clinical policy、输入/ROI QC 接入默认 Tool 注册表 | `config.py`、`schemas.py`、`imaging.py`、`segmentation.py`、`classification.py`、`workflow.py`、`verification.py`、测试与 README | 服务器 Agent 回归 4/4；真实 LangGraph 六步链和 FastAPI `/run` 均通过；概率与独立推理一致 | 分类 AUC/CI 尚不足临床准入；标签语义未确认；RAG/Report/Verifier 尚未接 LLM | 接入 Qwen structured output，并建立真实模型 Agent Benchmark |

## 八、阶段交付模板

每完成一个阶段，复制并填写以下内容：

```markdown
### Phase X 交付记录

- 状态：待验收
- 完成日期：YYYY-MM-DD
- 本阶段目标：
- 实际完成内容：
- 修改文件：
- 运行命令：
- 测试结果：
- 性能或质量指标：
- 验收证据：
- 未验证内容：
- 已知限制：
- 新增风险：
- 是否满足进入下一阶段的条件：是/否
- 下一阶段建议：
```

### Phase 0 交付记录

- 状态：待验收
- 完成日期：2026-07-31
- 本阶段目标：审计现有腰椎分割仓库，确认真实模型、数据、推理与风险边界，并形成最小侵入改造方案。
- 实际完成内容：完成 10 项当前执行命令；交付技术栈/目录、模型与入口、数据与预处理、checkpoint、类别/mask、复用清单、风险、工具映射、最小改造和 Phase 1 验收方案。
- 修改文件：`PHASE0_REPOSITORY_AUDIT.md`、`SPINEVISION_AGENT_PROJECT.md`
- 运行命令：`rg --files`、定向 `rg` 审计、Python AST 解析、checkpoint `weights_only=True` 元数据读取、SHA-256、合成张量前向和合成患者拆分检查。
- 测试结果：16/16 个 Python 文件通过 AST；4/4 类别契约测试通过；`last.pt` 输出 `1×13×256×256`；`best.pt` 输出 `1×13×640×640`；合成拆分复现 train/validation 患者交集。
- 性能或质量指标：仅确认旧 checkpoint 记录；未将旧 mean Dice 认定为患者级或独立测试性能。
- 验收证据：`PHASE0_REPOSITORY_AUDIT.md` 第 12 节。
- 未验证内容：真实患者数据端到端、`1..12` 正式标签名称/顺序、底层模态/序列、患者级性能、外部/时间外性能、DICOM/PNG 模型适用性、医学规则有效性。
- 已知限制：仓库不含原始数据、正式标签字典、自动化测试、独立测试结果和数据治理信息。
- 新增风险：R-13～R-16。
- 是否满足进入下一阶段的条件：部分满足；类别数量、NIfTI 训练格式和 `best.pt` 基线已确认，仍需正式标签名称/顺序、底层模态/序列和患者级拆分信息。若暂时无法确认，可仅以 `class_N` 和研究模式实现技术适配。
- 下一阶段建议：按审计报告第 11 节实施 `SegmentationService`、Pydantic schema、原空间恢复、基础 QC、patient-level split 与回归测试。

## 九、项目完成后的简历与面试材料

项目完成后输出：

- 150～250 字中文项目介绍；
- 4～6 条简历 bullet，包含技术、问题、行动和真实指标；
- 5 分钟面试讲解：临床问题、Agent 必要性、工具抽象、长程规划、Verifier、Benchmark、Agentic RL、限制和下一步；
- 单 Agent、多 Agent、RAG、Verifier 和后训练的消融实验结果；
- 未实测指标不得使用虚构数字，统一标记为“待实测”。
