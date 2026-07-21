# 基于图像的类器官类型与亚型/组别分类原型框架

本项目当前方向是构建一个可扩展的 organoid image classification framework，而不是一次性完成“所有类器官通用亚型识别”。

当前项目目标：

- 整理类器官图像、标签、mask、bbox 和实验 metadata。
- 在已有三组数据上验证图像分类 pipeline。
- 以肠道类器官形态亚型分类作为主科学任务。
- 将脑类器官作为 clone/group 分类辅助任务，而不是强行称为形态亚型分类。
- 为未来新类器官数据集预留“登记 metadata、少量人工标注、微调、验证”的扩展流程。

## 当前任务定义

### 1. 主任务：肠道类器官形态亚型分类

数据来源：

- 人小肠类器官 crops
- 小鼠小肠类器官 YOLO object crops

统一标签：

- `cyst`
- `early_budding`
- `late_budding`
- `spheroid`

这是当前最重要、最适合写成研究主线的任务。

### 2. 辅助任务：脑类器官 clone/group 分类

数据来源：

- 脑类器官图像

标签：

- `A1A-1`
- `B2A-2`
- `TH2-7`
- `wt2D`

注意：这些标签来自 clone/group metadata，不是已经人工确认的 morphology subtype。

### 3. 框架任务：类器官 domain/type 判断

标签：

- `human_intestine`
- `mouse_intestine`
- `brain`

这个任务用于 domain routing、质量控制和未来数据集扩展。

完整标签体系和训练清单见：

- `metadata/final_label_schema.md`
- `metadata/training_manifest.csv`

## 当前实验结果

目前已经完成四组小肠类器官形态亚型四分类实验，统一预测：

```text
cyst / early_budding / late_budding / spheroid
```

| Run | 数据 | 设置 | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: |
| `intestine_subtype_b0` | 人小肠 + 小鼠小肠 | weighted sampler | `0.9227` | `0.9218` | `0.9099` |
| `human_intestine_b0` | 仅人小肠 | weighted sampler | `0.8907` | `0.8958` | `0.8766` |
| `mouse_intestine_b0` | 仅小鼠小肠 | weighted sampler | `0.8866` | `0.8916` | `0.8667` |
| `intestine_subtype_b0_no_sampler` | 人小肠 + 小鼠小肠 | no weighted sampler | `0.9251` | `0.9177` | `0.9102` |
| `mil_multitask_mobile` | 人小肠 + 脑 + 小鼠小肠 | Attention-MIL multitask, MobileNetV3-small | `0.5019` | `0.4996` | `0.4479` |

主要结论：

- 人小肠 + 小鼠小肠联合训练优于单数据域训练，说明两组肠道类器官图像存在可共享的形态特征。
- 当前四分类任务具有可行性，联合模型达到约 `92%` validation accuracy 和 `91%` macro-F1。
- weighted sampler 对最终结果影响较小；启用和不启用时结果非常接近。
- 主要混淆来自 `early_budding` vs `late_budding` 和 `cyst` vs `spheroid`，符合类器官形态连续变化的特点。
- Attention-MIL 多任务 mobile 对照没有超过 crop/object-level expert baseline。当前小肠数据已有较好的 crop 或 bbox，因此直接训练 object/crop-level expert model 更适合作为主结果。

详细记录见：

- `EXPERIMENT_RESULTS.md`
- `metadata/experiment_results.csv`
- `logs/*/result_summary.md`

## 当前训练路线

本项目包含四条路线：

1. **单数据集 expert model**：分别训练人小肠、小鼠小肠、脑类器官模型，用于提高单域表现。
2. **肠道统一四分类模型**：只使用人小肠 + 小鼠小肠，预测 `cyst / early_budding / late_budding / spheroid`。
3. **Attention-MIL 多任务训练**：共享 backbone，并为不同数据域设置独立 head，用作跨数据源对照实验。
4. **未知亚型发现**：当不知道亚型数量或名称时，先聚类、人工观察代表图片，再命名并训练分类器。

当前最推荐的研究顺序：

```text
metadata 整理
→ 肠道四分类 expert baseline
→ Attention-MIL 多任务对照
→ 错误分析和混淆矩阵
→ 新数据扩展流程说明
```

---

## 项目结构

```text
organoid_project/
├── config.py                         # 单数据集训练参数
├── dataset.py                        # 单数据集加载与增强
├── model.py                          # 单数据集分类模型
├── train.py                          # 单数据集训练脚本
├── train_binary.py                   # 二分类训练脚本
├── evaluate.py                       # 单数据集评估脚本
├── multi_dataset.py                  # 三数据集加载器
├── multi_model.py                    # 三数据集普通多任务模型
├── multi_train.py                    # 三数据集普通多任务训练
├── mil_model.py                      # Attention-MIL 多任务模型
├── train_mil_multitask.py            # 推荐：Attention-MIL 多任务训练
├── build_metadata_manifest.py        # 生成统一图像/对象级 metadata 数据库
├── build_training_manifest.py        # 生成最终训练 manifest
├── discover_subtypes.py              # 未知亚型发现：特征提取 + 降维 + 聚类
├── export_misclassified_samples.py   # 导出误分类样本和困难样本图片面板
├── organoid_ontology.py              # 类器官类型/亚型/形态特征知识库
├── MIL_README.md                     # MIL 方案详细说明
├── data/                             # 脑类器官数据
├── metadata/                         # 统一图像/对象级 metadata manifest
├── models/                           # 模型权重，不上传 GitHub
└── logs/                             # 指标与图表
```

三组数据默认路径：

```text
/Users/zhaozhihe/Desktop/2026surf/Final_Organoids_Dataset人小肠
/Users/zhaozhihe/Desktop/2026surf/organoid_project/data
/Users/zhaozhihe/Desktop/2026surf/OrganoidDataset小鼠小肠
```

---

## 统一 Metadata 数据库

项目现在包含一个可重复生成的核心 metadata 数据库，用来连接图像、标签、对象框、mask 和实验背景。

重新生成：

```bash
python build_metadata_manifest.py
```

输出位置：

```text
metadata/
├── image_manifest.csv                # 每张图/裁剪图一行
├── object_manifest.csv               # 每个类器官实例一行
├── image_manifest_summary.csv        # 图像级标签分布
└── object_manifest_summary.csv       # 对象级标签分布
```

当前 manifest 覆盖三组数据：

- 人小肠 CLORG：文件夹类别映射到 `cyst`、`early_budding`、`late_budding`、`spheroid`。
- 脑类器官 OrganoIDNet：连接 `dataset_overview.csv`、原图、mask、day、clone、imaging source；当前标签来自 clone metadata。
- 小鼠小肠 OrgaQuant：图像级记录整图，对象级记录 YOLO bbox 和类别标签。

注意：脑类器官当前使用 clone 作为监督标签，这对实验可用，但不等同于已经人工确认的视觉形态亚型。

---

## 环境安装

```bash
cd /Users/zhaozhihe/Desktop/2026surf/organoid_project
pip install -r requirements.txt
```

如果只想手动安装核心依赖：

```bash
pip install torch torchvision pandas numpy scikit-learn opencv-python pillow matplotlib tqdm
```

检查 PyTorch 是否能用 GPU/MPS：

```bash
python - <<'PY'
import torch
print('CUDA:', torch.cuda.is_available())
print('MPS:', torch.backends.mps.is_available())
PY
```

---

## 推荐方案：Attention-MIL 多任务训练

### 为什么推荐

你的项目有三个典型困难：

- 样本数量有限，直接训练大型 CNN 容易过拟合。
- 三个数据集来源不同，标签含义并不完全一样。
- 切割步骤被弱化，整张图中可能含有背景、噪声或多个区域。

Attention-MIL 的做法是：

```text
整张图片
→ 切成多个 patch
→ 每个 patch 提取特征
→ attention 自动判断哪些 patch 重要
→ 输出该数据集对应的亚型分类
```

模型结构：

```text
共享 backbone
├── tissue head：判断数据来源/组织类型
├── human_intestine subtype head
├── brain subtype head
└── mouse_intestine subtype head
```

这样既能利用三组数据共享图像特征，又不会强行把三组标签混成一个分类任务。

### 快速测试

用于确认代码和数据能跑通：

```bash
python train_mil_multitask.py \
  --epochs 1 \
  --max-samples 60 \
  --max-val-samples 60 \
  --image-size 128 \
  --patch-size 64 \
  --batch-size 2 \
  --no-pretrained \
  --run-name smoke
```

### CPU 可跑版本

如果当前环境只能用 CPU，建议先用轻量模型：

```bash
python train_mil_multitask.py \
  --backbone mobilenet_v3_small \
  --epochs 20 \
  --freeze-epochs 5 \
  --image-size 384 \
  --patch-size 128 \
  --max-samples 2000 \
  --max-val-samples 800
```

### 正式训练版本

如果有 GPU 或 MPS，推荐：

```bash
python train_mil_multitask.py \
  --backbone convnext_tiny \
  --epochs 30 \
  --freeze-epochs 5 \
  --image-size 384 \
  --patch-size 128 \
  --max-samples 0 \
  --max-val-samples 0
```

其中：

- `--max-samples 0` 表示使用完整训练集。
- `--max-val-samples 0` 表示使用完整验证集。
- `--freeze-epochs 5` 表示前 5 轮冻结 backbone，只训练分类头，之后再微调 backbone。
- 默认会启用每个数据集内部的类别权重，减少大类别主导训练；如果要关闭，添加 `--no-class-balance`。
- 如果要用已有 mask 对脑类器官图像先裁剪再切 patch，添加 `--use-brain-mask`。
- 如果只是测试或做消融实验，建议添加 `--run-name 名称`，避免覆盖默认模型和日志。

---

## 输出结果

MIL 训练会保存：

```text
models/mil_multitask_best.pth          # 验证 macro-F1 最好的模型
models/mil_multitask_last.pth          # 最后一轮模型
logs/mil_multitask_metrics.json        # 每轮指标、每组数据指标、混淆矩阵
```

训练时重点看这些指标：

- `val_F1`：三组数据平均 macro-F1，最重要。
- `val_BalAcc`：三组数据平均 balanced accuracy。
- 每个数据集自己的 `F1` 和 `BalAcc`。
- `tissue_acc`：模型是否能区分三组数据来源。

不要只看普通 accuracy。样本不均衡时，accuracy 可能会误导。

---

## 单数据集 baseline

如果只训练脑类器官数据集：

```bash
python train.py
```

该脚本使用：

- `config.py` 中的参数
- `dataset.py` 的单数据集加载器
- `model.py` 的分类模型
- 5-fold group cross validation

模型保存到：

```text
models/best_model_fold1.pth
models/best_model_fold2.pth
...
```

评估：

```bash
python evaluate.py
```

---

## 普通三数据集多任务训练

如果想不用 MIL，直接训练三组数据：

```bash
python multi_train.py
```

这个版本会直接把图像 resize 后输入模型。它可以作为对照实验，但在切割不精确时，推荐优先使用 `train_mil_multitask.py`。

---

## 未知亚型发现

如果你不知道类器官有几种亚型，也不知道亚型名称，不应该直接训练普通分类器。更合理的流程是：

```text
图片
→ 预训练模型提取 embedding
→ PCA/UMAP 降维
→ DBSCAN/HDBSCAN/KMeans 聚类
→ 导出每个 cluster 的代表图片
→ 人工观察并命名 cluster
→ 再训练分类器
```

运行内置三组数据的未知亚型发现：

```bash
python discover_subtypes.py \
  --dataset all \
  --max-samples 1000 \
  --backbone mobilenet_v3_small
```

对任意未知图片文件夹运行：

```bash
python discover_subtypes.py \
  --input-dir /path/to/unknown_organoid_images \
  --max-samples 0
```

输出位置：

```text
logs/discovery/
├── embeddings.npy                    # 每张图的高维特征
├── reduced_2d.npy                    # 2D 降维坐标
├── cluster_assignments.csv           # 每张图属于哪个 cluster
├── clusters_2d.png                   # 聚类可视化图
├── summary.json                      # 每个 cluster 的统计信息
└── cluster_examples/                 # 每个 cluster 的代表图片
```

如果安装了 `umap-learn` 和 `hdbscan`，脚本会优先使用 UMAP + HDBSCAN；如果没有安装，会自动退化为 PCA + DBSCAN。

可选增强依赖安装：

```bash
pip install -r requirements-discovery.txt
```

### 类器官亚型知识库

项目内置了一个初步知识库：`organoid_ontology.py`。它整理了常见类器官系统、亚型/形态名称和显微图像中可能观察到的特征，例如：

- 通用形态：`spheroid`、`cystic`、`budding`、`solid_compact`、`branched`。
- 肠道：`enterosphere`、`early_budding`、`late_budding`、`cystic_intestinal`。
- 脑/神经：`cerebral_organoid`、`forebrain_organoid`、`midbrain_organoid`、`assembloid`。
- 肝胆/胰腺/胃/肺/肾/视网膜等常见系统。

单独导出知识库：

```bash
python organoid_ontology.py
```

运行 `discover_subtypes.py` 时，会自动把候选知识库导出到结果目录：

```text
logs/discovery/organoid_ontology.csv
logs/discovery/organoid_ontology.json
```

注意：这个知识库用于辅助解释 cluster 和设计标签体系，不能替代人工标注，也不能证明某张图一定属于某个亚型。

---

## 常见问题

**Q: 为什么三组数据一起训练不一定比单数据集高？**

A: 三组数据来自不同组织、物种或成像条件，标签语义也不同。简单合并可能让模型学到“数据集差异”，而不是“类器官亚型”。所以推荐多任务结构，每组数据用自己的分类头。

**Q: 为什么推荐 macro-F1 和 balanced accuracy？**

A: 因为类别可能不均衡。普通 accuracy 容易被样本多的类别主导，macro-F1 和 balanced accuracy 更能反映每个亚型是否都学得好。

**Q: 为什么弱化切割后要用 MIL？**

A: 如果整张图里有背景或多个区域，普通 CNN 会被无关背景干扰。MIL 会把图切成 patch，并用 attention 自动关注重要区域。

**Q: 训练很慢怎么办？**

A: 优先确认是否有 CUDA/MPS。如果只能 CPU，先用：

```bash
python train_mil_multitask.py \
  --backbone mobilenet_v3_small \
  --image-size 384 \
  --patch-size 128 \
  --max-samples 2000 \
  --max-val-samples 800
```

**Q: 想正式发表/写报告，应该展示什么？**

A: 建议展示：

- 每个数据集的 macro-F1 / balanced accuracy
- 每个数据集的 confusion matrix
- 单数据集 baseline vs 三数据集多任务 vs Attention-MIL
- 如果后续加入 attention 可视化，可以展示模型关注区域
