# 小鼠小肠类器官形态亚型四分类训练结果记录

记录时间：`2026-07-13 23:00:03 CST`

运行名称：`mouse_intestine_b0`

本次训练任务：小鼠小肠类器官形态亚型四分类

本次训练目标：

```text
输入：小鼠小肠类器官 YOLO bbox 裁剪图像
输出：cyst / early_budding / late_budding / spheroid 四种形态亚型
```

## 一、任务内容

本次训练只使用 `mouse_intestine` 数据域，不包含人小肠和脑类器官数据。

原始小鼠标签已经映射为统一四分类标签：

- `organoid0_cyst` → `cyst`
- `organoid1_early` → `early_budding`
- `organoid3_late` → `late_budding`
- `spheroid` → `spheroid`

## 二、模型与训练配置

| 项目 | 内容 |
| --- | --- |
| 模型 backbone | `efficientnet_b0` |
| 预训练权重 | 使用 ImageNet 预训练权重 |
| 训练数据域 | `mouse_intestine` |
| 训练轮数 | `25` epochs |
| 输入图像大小 | `224 x 224` |
| batch size | `32` |
| weighted sampler | 启用 |
| class-balanced loss | 启用 |
| bbox padding | `0.15` |
| 学习率 | `0.0003` |
| weight decay | `0.0001` |
| dropout | `0.3` |

## 三、最佳结果

最佳 epoch：`25`

| 指标 | 数值 |
| --- | ---: |
| Validation accuracy | `0.8866` |
| Validation balanced accuracy | `0.8916` |
| Validation macro-F1 | `0.8667` |
| Validation loss | `0.7193` |

可以表述为：

```text
本次小鼠小肠类器官四分类模型在验证集上达到约 88.66% accuracy 和 86.67% macro-F1。
```

## 四、每个类别的表现

| 类别 | Precision | Recall | F1 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| `cyst` | `0.9654` | `0.8842` | `0.9230` | `1295` |
| `early_budding` | `0.8017` | `0.8704` | `0.8346` | `548` |
| `late_budding` | `0.8747` | `0.9052` | `0.8897` | `401` |
| `spheroid` | `0.7473` | `0.9067` | `0.8193` | `225` |

## 五、混淆矩阵

行表示真实标签，列表示模型预测标签。

| True \ Pred | `cyst` | `early_budding` | `late_budding` | `spheroid` |
| --- | ---: | ---: | ---: | ---: |
| `cyst` | `1145` | `80` | `2` | `68` |
| `early_budding` | `21` | `477` | `50` | `0` |
| `late_budding` | `0` | `37` | `363` | `1` |
| `spheroid` | `20` | `1` | `0` | `204` |

主要混淆：

- `cyst` 被误分为 `early_budding`：`80` 个
- `cyst` 被误分为 `spheroid`：`68` 个
- `early_budding` 被误分为 `late_budding`：`50` 个
- `late_budding` 被误分为 `early_budding`：`37` 个
- `spheroid` 被误分为 `cyst`：`20` 个

## 六、结果解释

本次结果说明：在小鼠小肠类器官 YOLO bbox 裁剪图像上，EfficientNet-B0 可以较好地区分四种形态亚型。

表现最好的类别：

- `cyst`
- `late_budding`

相对较难的类别：

- `early_budding`
- `spheroid`

其中 `spheroid` 的 recall 较高，但 precision 较低，说明模型能找出大部分真实 `spheroid`，但也会把一部分其他类别误判为 `spheroid`。

## 七、输出文件

模型文件：

```text
models/mouse_intestine_b0_best.pth
models/mouse_intestine_b0_last.pth
```

结果文件：

```text
logs/mouse_intestine_b0/metrics.json
logs/mouse_intestine_b0/confusion_matrix.png
logs/mouse_intestine_b0/result_summary.md
```

## 八、当前结论

本次训练可以作为小鼠小肠单数据域 expert baseline。

中文表述：

```text
小鼠小肠类器官形态亚型四分类模型在验证集上取得了 88.66% 的准确率和 86.67% 的 macro-F1，说明单独在小鼠小肠数据域内进行形态亚型分类具有较好的可行性。
```
