# 类器官亚型分类 —— 训练教程

## 项目结构

```
organoid_project/
├── data/                       # 数据集
│   ├── dataset_overview.csv    # 图像清单 (1407 张)
│   ├── imgs/                   # 原始明场图像 (.jpg / .tif)
│   └── labels/                 # 分割标注 (.npy)
├── config.py                   # 所有参数（改这里调超参）
├── dataset.py                  # 数据加载 + 自动裁剪
├── model.py                    # 模型架构
├── train.py                    # 训练主脚本
├── evaluate.py                 # 评估 + 可视化
├── models/                     # 保存的模型权重
└── logs/                       # 评估图表
```

## 快速开始

### 1. 确认环境

```bash
# Python 3.10+ 且已安装依赖
pip install torch torchvision pandas numpy scikit-learn opencv-python pillow matplotlib tqdm
```

### 2. 确认数据就位

```bash
# 应该看到 3 个东西
ls data/
# 输出: dataset_overview.csv  imgs/  labels/

# 1407 张图像, 1407 个 mask
ls data/imgs/ | wc -l    # → 1407
ls data/labels/ | wc -l  # → 1407
```

### 3. 一行训练

```bash
cd ~/Desktop/2026surf/organoid_project
python train.py
```

训练过程中你会看到：
- `📊 数据集概况` — 数据统计
- `🔀 Fold N` — 第 N 折交叉验证
- 每轮打印 loss / acc / f1 / auc
- 最佳模型保存到 `models/best_model_foldN.pth`

### 4. 查看结果

```bash
python evaluate.py
```

---

## 怎么调参（改 config.py）

如果准确率不够 75%，按顺序试：

| 参数 | 默认值 | 调大 | 调小 | 说明 |
|------|:---:|------|------|------|
| `IMAGE_SIZE` | 224 | 384 或 512 | - | 分辨率越大，细节越多（但更慢） |
| `BACKBONE` | `efficientnet_b3` | `convnext_base` | `resnet50` | 换模型架构试试 |
| `DROPOUT` | 0.3 | 0.5 | 0.1 | 过拟合就调大 |
| `LEARNING_RATE` | 1e-4 | 5e-4 | 1e-5 | 收敛慢就调大 |
| `FREEZE_STAGES` | 3 | 0 | 5 | 冻结层数越少，训练越充分 |

## 训练目标

- 4 分类随机基线: **25%**
- 目标: **>75%**
- 当前最佳: **~69%** (Fold 2)

## 常见问题

**Q: 训练很慢？**
A: Mac MPS 每轮 ~60-150 秒属正常。可以设 `IMAGE_SIZE=128` 快速测试。

**Q: Out of Memory？**
A: 设 `BATCH_SIZE=16` 或 `8`。

**Q: 想用 GPU 服务器？**
A: config.py 自动检测 CUDA/MPS/CPU，代码不需要改。
