# SpineVision-Agent 工程 Benchmark 报告

> 测试日期：2026-08-07  
> 原始结果：`agent_artifacts/benchmark/benchmark_results.json`  
> 适用范围：工程工作流验证，不是临床性能、诊断准确率或真实医生效率研究。

## 1. 测试结论

| 指标 | 实测结果 | 测试条件 |
|---|---:|---|
| 自动化 Agent 数量 | 6 | Planner、Segmentation、Diagnosis、RAG、Report、Verifier |
| 人工复核节点 | 1 | Human Review，不计入自动 Agent |
| 注册 Tool 数量 | 6 | 输入检查、分割、分类、检索、报告、校验 |
| 正常病例 Tool 调用次数 | 6.0 次/例 | 30 个 mock 编排病例及 5 个真实分割病例均为 6 次 |
| 完整流程平均耗时 | 2.02 秒/例 | 5 例合成 NIfTI；每例 3 切片；真实 `best.pt` CPU 分割；分类为 mock |
| 完整流程中位耗时 | 1.41 秒/例 | 同上 |
| 完整流程 P95 | 3.89 秒/例 | 同上，包含首次模型加载 |
| 热启动平均耗时 | 1.39 秒/例 | 排除首次 4.51 秒冷启动后，其余 4 例平均值 |
| RAG Recall@1 | 0.6875 | 8 个标注查询、4 篇内部 SOP |
| RAG Recall@2 | 0.7500 | 同上 |
| RAG Recall@3 | 0.8750 | 同上；Hit Rate@3 = 1.000 |
| RAG Recall@4 | 1.0000 | Top-4 等于召回全部 4 篇文档，参考价值有限 |
| RAG MRR | 0.9167 | 8 个标注查询 |
| 报告结构化字段正确率 | 100/100，100% | Wilson 95% CI：96.3%–100%；仅衡量 schema、字段类型和证据引用 |
| Verifier 不一致检出率 | 120/120，100% | Wilson 95% CI：96.9%–100%；6 类合成不一致 |
| Verifier 干净样本通过率 | 20/20，100% | Wilson 95% CI：83.9%–100% |
| 异常恢复成功率 | 50/50，100% | Wilson 95% CI：92.9%–100%；5 个阶段各注入 10 次单次瞬时故障 |
| 人工操作步骤 | 8 步 → 3 步 | 按预定义流程计数，减少 62.5%；未进行真实医生计时研究 |

## 2. Agent 与 Tool

自动化 Agent：

1. Planner：解析目标并生成受限 ReAct 执行计划；
2. Segmentation：调用真实分割模型并输出 ROI/QC；
3. Diagnosis：调用分类后端并输出概率和不确定性；
4. RAG：检索版本化医学证据；
5. Report：生成带证据引用的结构化报告草稿；
6. Verifier：检查置信度、节段、数值、诊断类别和证据一致性。

注册 Tool：

```text
inspect_mri_input
run_spine_segmentation
run_spine_diagnosis
retrieve_medical_knowledge
generate_grounded_report
verify_case_consistency
```

正常病例每个 Tool 调用一次，因此平均为 6 次。注入一次可恢复故障时，对应 Tool 重试一次，总调用次数为 7 次。

## 3. 完整流程耗时

真实分割工作流使用 `runslite_256/best.pt`、CPU、5 个合成三切片 NIfTI 病例；分类仍使用 mock 后端。5 次耗时为：

```text
4.506 s, 1.364 s, 1.388 s, 1.415 s, 1.410 s
```

首次运行包含模型加载，后续复用模型缓存。该结果不能直接代表真实临床 MRI：真实病例切片数、磁盘、GPU、分类模型和 LLM/API 延迟均会改变总耗时。

## 4. RAG 召回

RAG 测试集包含 8 个查询，每个查询预先标注 1–2 个相关 evidence ID；语料库当前只有 4 篇内部安全 SOP。

| K | Mean Recall@K | Hit Rate@K |
|---:|---:|---:|
| 1 | 0.6875 | 0.8750 |
| 2 | 0.7500 | 0.8750 |
| 3 | 0.8750 | 1.0000 |
| 4 | 1.0000 | 1.0000 |

当前建议使用 `top_k=3`：所有查询至少命中一篇相关文档，同时减少无关上下文。扩大到正式指南库后，必须重新建立独立检索测试集；当前 Top-4 的 100% 是小语料库下的全量召回，不代表真实 RAG 水平。

## 5. 报告结构化字段

对 100 个不同置信度组合运行 Report Tool，检查：

- 必填字段完整；
- Pydantic 类型与枚举合法；
- Finding 引用的 evidence ID 存在；
- 分类结论保持 `inferred`，不升级为确定性诊断；
- 诊断类别与 Diagnosis Tool 输出一致；
- 报告状态保持 `draft`。

结果为 100/100 通过。这是结构化契约正确率，不是报告临床正确率、语言质量或医生一致率。

## 6. Verifier

每类注入 20 次，共 120 个不一致：

| 不一致类型 | 检出数/总数 |
|---|---:|
| 低分类置信度 | 20/20 |
| 引用不存在的 evidence ID | 20/20 |
| 标签语义未验证却生成具体节段 | 20/20 |
| 报告前景体素数与分割输出不一致 | 20/20 |
| 报告诊断类别与分类输出不一致 | 20/20 |
| 推断性结论缺少检索证据 | 20/20 |

另外 20 个无注入错误的样本均通过。该测试衡量已编码规则对已知错误模式的覆盖率，不代表 Verifier 能发现未知临床错误。

## 7. 异常恢复

在以下五个阶段分别注入 10 次“首次失败、第二次成功”的可恢复故障：

- 输入检查；
- 分割；
- 分类；
- RAG；
- 报告生成。

50 个病例均在一次重试后恢复并继续进入 Verifier，成功率为 100%（Wilson 95% CI：92.9%–100%）。当前结果仅适用于单次瞬时故障；持续故障、GPU OOM、损坏 NIfTI 和外部服务超时仍需单独测试。

## 8. 人工步骤

人工基线定义为 8 步：导入 MRI、启动分割、检查/导出 ROI、导入分类程序、启动分类、检索知识、撰写报告、核对确认。

Agent 流程定义为 3 步：提交 MRI、查看聚合结果和告警、接受/修改/驳回报告。

因此按流程计数由 8 步降至 3 步，减少 62.5%。这不是观察性时间动作研究；要写入临床论文或正式产品材料，仍需开展医生用户研究并记录真实操作次数和耗时。

## 9. 复现命令

```powershell
& 'D:\Application\CodeEditor\anaconda3\envs\pytorch\python.exe' `
  '.\run_agent_benchmark.py' `
  --output-dir '.\agent_artifacts\benchmark' `
  --real-cases 5 `
  --real-slices 3
```
