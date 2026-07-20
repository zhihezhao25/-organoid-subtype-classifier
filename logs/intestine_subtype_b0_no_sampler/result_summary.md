# 小肠类器官形态亚型四分类训练结果记录

记录时间：`2026-07-21 02:10:03 CST`

运行名称：`intestine_subtype_b0_no_sampler`

本次训练任务：小肠/肠道类器官形态亚型四分类

本次训练目标：

```text
输入：人小肠和小鼠小肠类器官图像或目标裁剪图
输出：cyst / early_budding / late_budding / spheroid 四种形态亚型
```

## 一、任务内容

本次训练是对主 baseline `intestine_subtype_b0` 的对照实验。

与主 baseline 相同，本次使用的数据域为：

- `human_intestine`：人小肠类器官 crop 图像
- `mouse_intestine`：小鼠小肠类器官 YOLO bbox 裁剪图像

统一分类标签：

- `cyst`
- `early_budding`
- `late_budding`
- `spheroid`

本次实验的关键区别是：不启用 `weighted_sampler`，用于观察样本加权采样对模型表现的影响。

## 二、模型与训练配置

| 项目 | 内容 |
| --- | --- |
| 模型 backbone | `efficientnet_b0` |
| 预训练权重 | 使用 ImageNet 预训练权重 |
| 训练数据域 | `both`，即人小肠 + 小鼠小肠 |
| 训练轮数 | `25` epochs |
| 输入图像大小 | `224 x 224` |
| batch size | `32` |
| weighted sampler | 未启用 |
| class-balanced loss | 启用 |
| 学习率 | `0.0003` |
| weight decay | `0.0001` |
| dropout | `0.3` |

## 三、最佳结果

最佳 epoch：`24`

| 指标 | 数值 |
| --- | ---: |
| Validation accuracy | `0.9251` |
| Validation balanced accuracy | `0.9177` |
| Validation macro-F1 | `0.9102` |
| Validation loss | `0.7315` |

可以表述为：

```text
本次不启用 weighted sampler 的小肠类器官四分类模型在验证集上达到约 92.51% accuracy 和 91.02% macro-F1。
```

## 四、每个类别的表现

| 类别 | Precision | Recall | F1 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| `cyst` | `0.9592` | `0.9487` | `0.9539` | `2357` |
| `early_budding` | `0.8960` | `0.8669` | `0.8812` | `1044` |
| `late_budding` | `0.8945` | `0.9369` | `0.9152` | `697` |
| `spheroid` | `0.8640` | `0.9184` | `0.8904` | `429` |

## 五、混淆矩阵

行表示真实标签，列表示模型预测标签。

| True \ Pred | `cyst` | `early_budding` | `late_budding` | `spheroid` |
| --- | ---: | ---: | ---: | ---: |
| `cyst` | `2236` | `59` | `2` | `60` |
| `early_budding` | `62` | `905` | `75` | `2` |
| `late_budding` | `0` | `44` | `653` | `0` |
| `spheroid` | `33` | `2` | `0` | `394` |

主要混淆：

- `early_budding` 被误分为 `late_budding`：`75` 个
- `cyst` 被误分为 `spheroid`：`60` 个
- `cyst` 被误分为 `early_budding`：`59` 个
- `late_budding` 被误分为 `early_budding`：`44` 个
- `spheroid` 被误分为 `cyst`：`33` 个

## 六、与主 baseline 的比较

主 baseline `intestine_subtype_b0` 启用了 `weighted_sampler`，结果为：

| Run | Weighted sampler | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| `intestine_subtype_b0` | 启用 | `0.9227` | `0.9218` | `0.9099` |
| `intestine_subtype_b0_no_sampler` | 未启用 | `0.9251` | `0.9177` | `0.9102` |

本次结果说明：

- 不启用 `weighted_sampler` 后，整体 accuracy 和 macro-F1 略有上升；
- 但 balanced accuracy 略低于主 baseline；
- 两组结果非常接近，说明当前模型表现主要来自图像分类 backbone 和 class-balanced loss，`weighted_sampler` 的影响不大；
- 如果报告中只选一个主结果，仍建议把 `intestine_subtype_b0` 作为主 baseline，把本次实验作为 ablation/control experiment。

## 七、输出文件

模型文件：

```text
models/intestine_subtype_b0_no_sampler_best.pth
models/intestine_subtype_b0_no_sampler_last.pth
```

结果文件：

```text
logs/intestine_subtype_b0_no_sampler/metrics.json
logs/intestine_subtype_b0_no_sampler/confusion_matrix.png
logs/intestine_subtype_b0_no_sampler/result_summary.md
```

## 八、当前结论

本次训练可以作为 weighted sampler 消融实验。

中文表述：

```text
在人小肠和小鼠小肠联合数据上，不启用 weighted sampler 的 EfficientNet-B0 模型取得了 92.51% 的验证集准确率和 91.02% 的 macro-F1。与启用 weighted sampler 的主 baseline 相比，整体表现非常接近，说明当前分类框架较稳定，但类别均衡策略对不同指标有轻微影响。
```
