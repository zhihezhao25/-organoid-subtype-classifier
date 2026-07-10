# 类器官亚型分类项目

本项目用于类器官亚型分类，包含三条训练路线：

1. **单数据集分类**：主要用于脑类器官数据集的 baseline。
2. **三数据集多任务训练**：同时使用人小肠、脑类器官、小鼠小肠三组数据。
3. **Attention-MIL 多任务训练（推荐）**：适合样本不足、数据来源不同、切割步骤弱化的情况。

当前最推荐使用第 3 条路线：`train_mil_multitask.py`。

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
├── MIL_README.md                     # MIL 方案详细说明
├── data/                             # 脑类器官数据
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
  --no-pretrained
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
