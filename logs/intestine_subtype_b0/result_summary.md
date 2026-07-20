# 小肠类器官形态亚型四分类训练结果记录

记录时间：`2026-07-13 11:13:56 CST`

运行名称：`intestine_subtype_b0`

本次训练任务：小肠/肠道类器官形态亚型四分类

本次训练目标：

```text
输入：人小肠和小鼠小肠类器官图像或目标裁剪图
输出：cyst / early_budding / late_budding / spheroid 四种形态亚型
```

## 一、任务内容

本次训练是当前项目的主科学任务，不包含脑类器官 clone/group 分类。

使用的数据域：

- `human_intestine`：人小肠类器官 crop 图像
- `mouse_intestine`：小鼠小肠类器官 YOLO bbox 裁剪图像

统一分类标签：

- `cyst`
- `early_budding`
- `late_budding`
- `spheroid`

## 二、模型与训练配置

| 项目 | 内容 |
| --- | --- |
| 模型 backbone | `efficientnet_b0` |
| 预训练权重 | 使用 ImageNet 预训练权重 |
| 训练数据域 | `both`，即人小肠 + 小鼠小肠 |
| 训练轮数 | `25` epochs |
| 输入图像大小 | `224 x 224` |
| batch size | `32` |
| weighted sampler | 启用 |
| class-balanced loss | 启用 |
| 学习率 | `0.0003` |
| weight decay | `0.0001` |
| dropout | `0.3` |

## 三、最佳结果

最佳 epoch：`25`

| 指标 | 数值 |
| --- | ---: |
| Validation accuracy | `0.9227` |
| Validation balanced accuracy | `0.9218` |
| Validation macro-F1 | `0.9099` |
| Validation loss | `0.7377` |

可以表述为：

```text
本次小肠类器官四分类模型在验证集上达到约 92.27% accuracy 和 90.99% macro-F1。
```

## 四、每个类别的表现

| 类别 | Precision | Recall | F1 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| `cyst` | `0.9676` | `0.9258` | `0.9462` | `2357` |
| `early_budding` | `0.8681` | `0.9080` | `0.8876` | `1044` |
| `late_budding` | `0.9123` | `0.9397` | `0.9258` | `697` |
| `spheroid` | `0.8485` | `0.9138` | `0.8799` | `429` |

## 五、混淆矩阵

行表示真实标签，列表示模型预测标签。

| True \ Pred | `cyst` | `early_budding` | `late_budding` | `spheroid` |
| --- | ---: | ---: | ---: | ---: |
| `cyst` | `2182` | `103` | `2` | `70` |
| `early_budding` | `35` | `948` | `61` | `0` |
| `late_budding` | `1` | `41` | `655` | `0` |
| `spheroid` | `37` | `0` | `0` | `392` |

主要混淆：

- `cyst` 被误分为 `early_budding`：`103` 个
- `cyst` 被误分为 `spheroid`：`70` 个
- `early_budding` 被误分为 `late_budding`：`61` 个
- `late_budding` 被误分为 `early_budding`：`41` 个
- `spheroid` 被误分为 `cyst`：`37` 个

## 六、结果解释

本次训练结果说明：在当前人小肠和小鼠小肠数据上，基于 EfficientNet-B0 的图像分类模型可以较好地区分四种常见小肠类器官形态亚型。

目前表现最好的类别是：

- `cyst`
- `late_budding`

相对较难的类别是：

- `early_budding`
- `spheroid`

这些混淆是合理的，因为类器官形态并不是完全离散的类别，而可能存在连续变化。例如：

- `early_budding` 与 `late_budding` 之间存在发育阶段过渡；
- `cyst` 与 `spheroid` 在部分图像中可能都呈现较圆的外观。

## 七、输出文件

模型文件：

```text
models/intestine_subtype_b0_best.pth
models/intestine_subtype_b0_last.pth
```

结果文件：

```text
logs/intestine_subtype_b0/metrics.json
logs/intestine_subtype_b0/confusion_matrix.png
logs/intestine_subtype_b0/result_summary.md
```

## 八、当前结论

本次训练可以作为项目当前的主 baseline 结果。

推荐在报告或展示中写为：

```text
The intestinal organoid morphology subtype classifier achieved 92.27% validation accuracy and 90.99% macro-F1 across four classes: cyst, early_budding, late_budding, and spheroid.
```

中文表述：

```text
本项目的小肠类器官形态亚型四分类模型在验证集上取得了 92.27% 的准确率和 90.99% 的 macro-F1，说明基于图像的自动类器官形态亚型分类具有较好的可行性。
```
