# Attention-MIL 多任务训练方案

这是当前项目更推荐的主方案，适合“样本少、三组数据来源不同、切割步骤弱化”的类器官亚型分类。

## 为什么用这个方案

- **预训练 backbone**：减少小样本训练压力。
- **Attention-MIL**：一张图切成多个 patch，模型自动学习哪些区域重要，不强依赖精确分割。
- **多任务分类头**：三组数据共享图像特征，但每组数据有自己的亚型分类头，避免标签语义混乱。
- **Macro-F1 / Balanced Accuracy**：比普通 accuracy 更适合类别不均衡。

## 文件

- `mil_model.py`：Attention-MIL 模型。
- `train_mil_multitask.py`：三数据集多任务训练入口。
- `logs/mil_multitask_metrics.json`：训练指标和混淆矩阵。
- `models/mil_multitask_best.pth`：最佳模型。

## 快速测试

```bash
cd /Users/zhaozhihe/Desktop/2026surf/organoid_project
python train_mil_multitask.py --epochs 1 --max-samples 60 --max-val-samples 60 --no-pretrained
```

## 正式训练

如果有 GPU/MPS：

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

如果只能用 CPU，建议先用轻量 backbone：

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

## 重点看哪些指标

每轮会输出：

- `val_F1`：三组数据平均 macro-F1，最重要。
- `val_BalAcc`：三组数据平均 balanced accuracy。
- 每个数据集自己的 `F1` 和 `BalAcc`。
- `tissue_acc`：模型是否能区分三组数据来源。

不要只看普通 accuracy，样本不均衡时会误导。
