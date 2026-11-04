# SpineVision SwinUMamba Lite

## 项目简介

针对腰椎 NIfTI 医学影像二维分割任务，本项目构建了一个轻量化 SwinUMamba 风格训练与部署流程：通过 VSS 风格残差门控块、深度可分离卷积、GroupNorm 和 U 形跳跃连接，在保留分割能力的同时适配低资源 CPU 环境；系统支持 NIfTI 数据扫描、标签重编码、Dice+CE 联合训练、验证指标统计、TorchScript 导出及单个 NIfTI 文件 CPU 推理。

已有检查点记录显示，背景 + 12 个前景 mask（共 13 个输出通道）任务在 512 个训练切片和 128 个验证切片上，最佳 epoch 为 26，验证集 mean Dice 为 0.8879，验证损失为 0.1639。外部测试集性能和推理耗时需另行实测。

这个工程把你的 `paXXX/img/CaseXXX.nii.gz`、`paXXX/label/mask_caseXXX.nii.gz` 预处理逻辑接入了一个可训练/测试/导出的 2D 分割流程。

默认模型是 `lite`：纯 PyTorch、低内存、适合最终在 2GB CPU 上部署。`official` 入口保留了你给的 `SwinUMambaD.py`，但它依赖 `mamba_ssm/timm/nnunetv2`，不建议直接用于 2GB CPU 部署。

## 数据结构

```text
dataset/train/
  pa267/
    img/Case267.nii.gz
    label/mask_case267.nii.gz
  pa105/
    img/Case105.nii.gz
    label/mask_case105.nii.gz
```

标签重编码沿用你的逻辑：

- `--num-classes 2`：所有大于 0 的标签合并为前景。
- `--num-classes 3`：标签 1-6 编为 1，标签 >=7 编为 2。
- `--num-classes >=4`：默认保留原始标签直接训练；例如标签是 `0..12` 就用 `--num-classes 13`，标签是 `0..13` 就用 `--num-classes 14`。

## 训练

先检查标签范围：

```bash
python inspect_labels.py --data-dir C:\path\to\dataset\train
```

输出里的 `recommended_num_classes` 就是保留原始标签时应传给 `--num-classes` 的值。

低内存推荐配置：

```bash
python train.py ^
  --train-dir C:\path\to\dataset\train ^
  --out-dir runs\lite_256 ^
  --model lite ^
  --num-classes 13 ^
  --label-mode preserve ^
  --image-size 256 ^
  --base-channels 12 ^
  --batch-size 1 ^
  --workers 0 ^
  --num-threads 2
```

如果训练机器内存/GPU 更充足，可以用：

```bash
python train.py --train-dir C:\path\to\dataset\train --image-size 320 --base-channels 16 --augment --rotate 8 --shift 8 8
```

`best.pt` 和 `last.pt` 会保存在 `--out-dir`。

## 测试

```bash
python test.py ^
  --checkpoint runs\lite_256\best.pt ^
  --data-dir C:\path\to\dataset\test ^
  --out-dir runs\lite_256_test ^
  --device cpu ^
  --num-threads 2
```

如果要保存每个 slice 的预测：

```bash
python test.py --checkpoint runs\lite_256\best.pt --data-dir C:\path\to\dataset\test --save-predictions
```

## CPU 部署导出

```bash
python export_cpu.py ^
  --checkpoint runs\lite_256\best.pt ^
  --output runs\deploy\spinevision_lite_cpu.ts ^
  --num-threads 2
```

单个 NIfTI 推理：

```bash
python infer_cpu.py ^
  --model runs\deploy\spinevision_lite_cpu.ts ^
  --image C:\path\to\Case267.nii.gz ^
  --output runs\deploy\Case267_pred.nii.gz ^
  --image-size 256 ^
  --num-threads 2
```

2GB CPU 部署建议：

- 优先用 `--image-size 256`，必要时再试 320。
- `--base-channels 8` 或 `12` 更稳；`16` 精度可能更好但内存更高。
- 推理时 batch 固定为 1。
- 保持 `num_workers=0`，避免 Windows 下多进程额外占内存。
- 训练增强默认不启用 elastic deformation；它很耗 CPU 和内存。

## 官方 SwinUMambaD 入口

你给的 `SwinUMambaD.py` 已放在：

```text
spinevision/models/SwinUMambaD.py
```

如果安装了 `requirements-official.txt` 中的依赖，可以尝试：

```bash
python train.py --train-dir C:\path\to\dataset\train --model official --image-size 320 --official-base-dim 24
```

注意：官方 SwinUMambaD 仍然比 `lite` 重很多，也可能依赖不适合 CPU 的 selective scan 实现。
