# ROI-assisted patient-level spondylolisthesis classifier

该子项目复用已训练的 `SwinUMambaLite` 分割模型，对每个病例中央 3 张矢状位切片生成 13 类 ROI，并以病例为最小训练和评估单位完成腰椎滑脱二分类。

## Architecture

```text
Sagittal central 3 slices
        |
        +--> pretrained segmentation encoder --> raw features ----+
        |                                                         |
        +--> segmentation mask --> dilated ROI image              +--> gated fusion
                              --> shared encoder --> ROI features -+        |
        +--> mask label fractions --------------------------------+        v
                                                                    slice attention
                                                                          |
                                                                          v
                                                               patient-level logits
```

- 原图和 ROI 分支共享同一套分割 Encoder 权重，避免模型规模翻倍。
- 每个病例只产生一次 loss，不把切片当成独立病例。
- 使用 CSV 已给出的 patient-level `train/val/test` 划分。
- 验证集选择 checkpoint 和 Youden 阈值，测试集只做冻结后的最终评价。
- 输出 AUC、准确率、敏感度、特异度、F1、PPV、NPV、Brier、ECE 和病例 bootstrap 95% CI。

## Remote usage

```bash
cd /data1/code/gjj/cls_all/spine_roi_case_cls

CUDA_VISIBLE_DEVICES=6 conda run -n mamba_mul \
  python prepare_roi_cache.py --config config.json --device cuda:0

CUDA_VISIBLE_DEVICES=6 conda run -n mamba_mul \
  python train.py --config config.json --device cuda:0

conda run -n mamba_mul python audit_results.py \
  --summary runs/roi_case_v1/summary.json \
  --predictions runs/roi_case_v1/test_predictions.csv \
  --output runs/roi_case_v1/center_audit.json
```

单病例推理：

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n mamba_mul python infer.py \
  --checkpoint runs/roi_case_v1/best.pt \
  --patient-id CASE_ID \
  --device cuda:0
```

## Reproduced run

固定病例级划分为训练 528 例、验证 68 例、测试 216 例。以验证集 AUC 选择最佳 checkpoint，并在验证集确定阈值 `0.5647`；最佳模型位于 epoch 32，训练耗时约 83 秒。

| Dataset | AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 0.798 | 0.750 | 0.765 | 0.735 | 0.754 |
| Test | 0.819 | 0.745 | 0.722 | 0.769 | 0.739 |

测试 AUC 的病例级 bootstrap 95% CI 为 `0.759-0.872`。测试结果按中心重算后 AUC 为 `0.765/0.790/0.781/0.835`；前三个中心仅 18-26 例，应避免过度解读。完整指标见 `runs/summary.json` 和 `runs/center_audit.json`（本地同步目录；远程路径为 `runs/roi_case_v1/`）。

## Clinical safety layer

当前 checkpoint 的资格评估状态为 **`NOT_CLINICALLY_QUALIFIED`**。项目新增了训练域输入质控、ROI 质控、温度校准、双阈值拒识、HMAC 病例引用、模型/策略哈希和 JSONL 审计日志。资格评估未通过全部门槛时，推理入口始终强制返回 `RESEARCH_ONLY_MANDATORY_REVIEW`。

生成资格策略：

```bash
python qualify_model.py \
  --checkpoint runs/roi_case_v1/best.pt \
  --summary runs/roi_case_v1/summary.json \
  --val-predictions runs/roi_case_v1/val_predictions.csv \
  --test-predictions runs/roi_case_v1/test_predictions.csv \
  --clinical-config clinical_config.json \
  --output-policy runs/roi_case_v1/clinical_policy.json \
  --output-report runs/roi_case_v1/clinical_qualification_report.json
```

执行带审计的 fail-closed 推理：

```bash
export SPINE_AUDIT_SECRET='replace-with-a-secret-from-your-secret-manager'
python clinical_infer.py \
  --checkpoint runs/roi_case_v1/best.pt \
  --policy runs/roi_case_v1/clinical_policy.json \
  --patient-id CASE_ID \
  --device cuda:0 \
  --output runs/roi_case_v1/clinical_case.json \
  --audit-log runs/roi_case_v1/clinical_audit.jsonl
```

完整临床证据缺口、MRMC 读者研究和前瞻性验证方案见 [`CLINICAL_READINESS.md`](CLINICAL_READINESS.md)。

## Data safety and interpretation

- 训练代码不修改原始 NIfTI 或标签表。
- `cache/` 只保存中央切片和模型生成的 Mask，可删除后重建。
- 分割标签目前只作为数值类别和前景 ROI 使用；在标签语义字典确认前，不赋予具体椎体/椎间盘名称。
- 分割预训练集 215 例与本次分类的 train/val/test 病例 ID 重叠均为 0。
- 本代码用于研究验证，不可直接用于临床诊断。
