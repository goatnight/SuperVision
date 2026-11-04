# SpineVision 临床指标标准、计算条件与实现状态

> 更新日期：2026-07-31  
> 适用版本：背景 1 类 + 前景 12 类 mask；默认基线 `runslite_256/best.pt`  
> 使用边界：研究辅助测量，不构成诊断、治疗建议或医疗器械性能声明。

## 1. 实施原则

临床指标只有在以下三项同时成立时才输出数值：

1. 当前模型的 mask 语义足以支持该指标；
2. 输入影像保留该指标所需的物理信息（方向、像素间距、真实强度标度等）；
3. 计算公式与已发表的临床测量定义一致。

任一项不满足，界面返回“不可计算：原因”，不使用替代灰度、包围盒比例或经验阈值猜测诊断。

## 2. 当前指标矩阵

| 指标 | 临床定义/可信参考 | 当前能否计算 | 当前实现 |
|---|---|---:|---|
| L1–S1 腰椎前凸角 | L1 上终板与 S1 上终板夹角；Cobb L1–S1 是常用总体腰椎前凸测量。[PubMed 37600414](https://pubmed.ncbi.nlm.nih.gov/37600414/)、[PubMed 29965893](https://pubmed.ncbi.nlm.nih.gov/29965893/) | **有条件可以** | 使用当前项目标签契约 label 1=L1、label 6=S1；NIfTI/DICOM 使用物理坐标，JPG/PNG 在矢状位且方形像素、未非等比拉伸的前提下可计算两终板相对夹角；不自动设置“正常/异常”阈值 |
| L4–S1 下腰椎前凸角 | L4 与 S1 上终板夹角，用于描述远端/下腰椎前凸分布。[PMC 6259763](https://pmc.ncbi.nlm.nih.gov/articles/PMC6259763/) | **有条件可以** | 要求正式标签字典、矢状位、物理坐标及合格的 L4/S1 上终板拟合；只输出角度，不自动解释失衡 |
| S1 骶骨倾斜角（sacral slope） | S1 上终板与水平线的夹角。[PMC 9993244](https://pmc.ncbi.nlm.nih.gov/articles/PMC9993244/)、[PMC 8965307](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965307/) | **有条件可以** | 使用患者物理前后轴作为水平线；明确标注体位相关，不把仰卧 MRI 数值直接套入站立位参考范围 |
| 椎管 mask 面积/体积 | 分割区域的物理面积/体积是确定性几何量，但不是狭窄分级 | **可以，描述性** | 单层且有二维间距时输出 mm²；三维且有三维间距时输出 cm³；明确标注“不得解释为狭窄分级” |
| 各类别 mask 面积/体积 | 体素计数乘物理体素尺寸 | **可以，描述性** | 保留在“结构量化报告”；无临床正常值或异常阈值 |
| 椎间盘高度/退变 | 可靠方法需要椎间盘轮廓或明确的终板标志点；面积法也以 disc contour 为基础。[PubMed 29432200](https://pubmed.ncbi.nlm.nih.gov/29432200/)、[PubMed 36758585](https://pubmed.ncbi.nlm.nih.gov/36758585/) | **当前不能** | 当前 12 类没有椎间盘 mask，也没有经过验证的终板标志点；已删除相邻包围盒间隙的 2.5%/7% 分级 |
| 腰椎管狭窄分级 | Schizas 分级依赖轴位 T2 上硬膜囊、神经根和脑脊液形态。[PubMed 20671589](https://pubmed.ncbi.nlm.nih.gov/20671589/)；WFNS 指南指出不存在单一、无歧义的量化影像标准，影像与症状也可能不一致。[PMC 7322797](https://pmc.ncbi.nlm.nih.gov/articles/PMC7322797/) | **当前不能** | 已删除标签 12/标签 11 总面积比的 55%/80% 分级；仅保留椎管 mask 几何量 |
| 骨密度/骨质疏松筛查 | 机会性 CT 可使用校准后的 L1 松质骨 HU ROI；DXA 是另一条标准路径。[PubMed 23588747](https://pubmed.ncbi.nlm.nih.gov/23588747/) | **当前不能** | 当前 NIfTI 训练域未确认是定量 CT；DICOM 被窗宽窗位归一化为 8 bit，PNG/MRI 无 CT HU。已删除“椎体灰度低于整图 0.75 倍”的规则 |
| 椎体压缩性骨折 | Genant 半定量评估需前/中/后缘椎体高度变化，并视觉排除非骨折性变形。[PubMed 8237484](https://pubmed.ncbi.nlm.nih.gov/8237484/)、[IOF 指南](https://www.osteoporosis.foundation/health-professionals/fragility-fractures/assessing-vertebral-fractures) | **当前不能自动分级** | 当前 mask 没有经验证的前/中/后缘标志点；即使未来得到高度比例，仍需人工形态学复核 |
| 脊柱侧弯 Cobb 角 | 标准 Cobb 角需要冠状位/正位影像上选定上下终椎的终板线 | **当前不能** | 原算法只是 L1–L5 质心连线相对纵轴的倾角，已删除且不再命名为 Cobb 角 |

## 3. 已实现公式

### 3.1 L1–S1 腰椎前凸角

1. 确认正式标签字典中 label 1=L1、label 6=S1；未确认时 fail-closed；
2. 从 NIfTI affine 或 DICOM `ImageOrientationPatient` 判断切片平面；
3. 将 mask 像素映射到患者物理坐标；
4. 在同时包含 L1 与 S1 的代表性矢状切片上，分别提取两者上缘的稳健包络；
5. 对 L1、S1 上终板分别拟合方向向量 `u`、`v`；
6. 计算锐角：

```text
angle = arccos(|u · v|), 0° ≤ angle ≤ 90°
```

当前不设置固定正常范围，原因包括体位（仰卧 MRI 与站立位侧位片）、骨盆参数、年龄和标志点选择均会影响结果。输出统一带“需医师复核”。

程序当前按 GUI/训练项目中的标签契约默认启用
`label 1=L1、label 4=L4、label 6=S1`。如果实际训练标签顺序不同，必须在启动前关闭：

```bat
set SPINE_LABEL_SEMANTICS_CONFIRMED=0
```

默认启用只代表程序采用该标签字典，不代表角度算法已经完成真实病例临床验证。

对于 JPG/PNG，L1–S1 和 L4–S1 是两条终板之间的相对夹角，在方形像素、
等比例缩放时不依赖毫米间距；结果会标注“JPG/PNG 方形像素假设”。
椎管面积/体积仍需要真实 spacing，S1 骶骨倾斜角仍需要可信患者水平轴。

### 3.2 L4–S1 下腰椎前凸角

计算过程与 L1–S1 相同，将上端终板替换为 L4 上终板。该指标描述
下腰椎前凸分布，不等同于独立疾病诊断。

### 3.3 S1 骶骨倾斜角

```text
sacral_slope = angle(S1 superior endplate, patient AP horizontal axis)
```

水平线来自 NIfTI affine 或 DICOM 患者坐标，而不是界面窗口的像素横轴。
仰卧 MRI 与站立侧位片的测量可能不同，因此当前不自动应用正常范围或手术规划阈值。

### 3.4 mask 物理面积/体积

```text
area_mm2   = mask_pixel_count × spacing_row_mm × spacing_column_mm
volume_cm3 = mask_voxel_count × spacing_x_mm × spacing_y_mm × spacing_z_mm / 1000
```

这是分割结果的描述性几何量。它不能单独证明椎管狭窄、疾病严重度或临床症状。

## 4. 若要继续增加可信临床指标，需要补充什么

| 目标指标 | 必需补充数据/模型输出 | 最小验证要求 |
|---|---|---|
| Schizas 狭窄分级 | 轴位 T2；节段定位；硬膜囊、脑脊液/神经根形态，或经验证的硬膜囊横截面积 | 双医师独立分级/一致性；锁定阈值后独立验证；同时报告与症状不一致的限制 |
| CT 骨质疏松筛查 | 原始 CT；`RescaleSlope/Intercept` 转 HU；L1 松质骨 ROI；排除皮质骨/静脉丛/局灶病变 | 与 DXA 或临床参考标准比较；扫描器/增强状态分层；外部验证 |
| 椎间盘高度 | L1/2–L5/S1 椎间盘 mask，或明确的终板角点/中点 | 与人工标准测量比较；ICC、Bland–Altman；按节段报告 |
| Genant 骨折分级 | T4–L4（或明确限定腰椎范围）的前/中/后缘标志点；原始侧位形态 | 人工视觉判读作为组成部分；与专家共识比较；不能只依赖高度比例 |
| Meyerding 滑脱分级 | 相邻椎体终板长度与后缘位移标志点；标准站立中立侧位片。[标准定义](https://pmc.ncbi.nlm.nih.gov/articles/PMC7170696/) | 明确位移/终板长度公式；观察者一致性；验证仰卧 MRI 与站立片差异。[体位一致性研究](https://pubmed.ncbi.nlm.nih.gov/31381804/) |
| 冠状位 Cobb 侧弯角 | 冠状位/正位全脊柱影像；上下终椎自动选择及终板线 | 与医师 Cobb 测量比较；终椎选择一致性；站立位验证 |

## 5. 代码对应关系

- 临床规则与 fail-closed 状态：`spinevision_clinical_metrics.py`
- 影像方向/间距元数据、界面展示与报告：`SpineVision_AI_real_infer.py`
- 单元测试：`tests/test_clinical_metrics.py`

当前界面不再输出自动治疗建议；“临床指标”区改为“标准化测量与可用性”，“治疗建议”区改为“复核与限制”。

暂未把 Meyerding 滑脱等级加入自动诊断。该等级的标准基于站立中立侧位片，
而仰卧 MRI 与站立片的滑移比例/等级一致性可能较差；在没有站立位标记、
后缘标志点验证和人工一致性实验前，只能作为后续候选，不能作为当前可靠结论。
