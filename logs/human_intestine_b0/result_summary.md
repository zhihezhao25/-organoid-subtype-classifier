# 人小肠类器官形态亚型四分类训练结果记录

记录时间：`2026-07-13 18:25:39 CST`

运行名称：`human_intestine_b0`

本次训练任务：人小肠类器官形态亚型四分类

本次训练目标：

```text
输入：人小肠类器官 crop 图像
输出：cyst / early_budding / late_budding / spheroid 四种形态亚型
```

## 一、任务内容

本次训练只使用 `human_intestine` 数据域，不包含小鼠小肠和脑类器官数据。

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
| 训练数据域 | `human_intestine` |
| 训练轮数 | `25` epochs |
| 输入图像大小 | `224 x 224` |
| batch size | `32` |
| weighted sampler | 启用 |
| class-balanced loss | 启用 |
| 学习率 | `0.0003` |
| weight decay | `0.0001` |
| dropout | `0.3` |

## 三、最佳结果

最佳 epoch：`22`

| 指标 | 数值 |
| --- | ---: |
| Validation accuracy | `0.8907` |
| Validation balanced accuracy | `0.8958` |
| Validation macro-F1 | `0.8766` |
| Validation loss | `0.8612` |

可以表述为：

```text
本次人小肠类器官四分类模型在验证集上达到约 89.07% accuracy 和 87.66% macro-F1。
```

## 四、每个类别的表现

| 类别 | Precision | Recall | F1 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| `cyst` | `0.9672` | `0.8889` | `0.9264` | `1062` |
| `early_budding` | `0.8056` | `0.8770` | `0.8398` | `496` |
| `late_budding` | `0.8508` | `0.9054` | `0.8773` | `296` |
| `spheroid` | `0.8194` | `0.9118` | `0.8631` | `204` |

## 五、混淆矩阵

行表示真实标签，列表示模型预测标签。

| True \ Pred | `cyst` | `early_budding` | `late_budding` | `spheroid` |
| --- | ---: | ---: | ---: | ---: |
| `cyst` | `944` | `77` | `1` | `40` |
| `early_budding` | `14` | `435` | `46` | `1` |
| `late_budding` | `0` | `28` | `268` | `0` |
| `spheroid` | `18` | `0` | `0` | `186` |

主要混淆：

- `cyst` 被误分为 `early_budding`：`77` 个
- `cyst` 被误分为 `spheroid`：`40` 个
- `early_budding` 被误分为 `late_budding`：`46` 个
- `late_budding` 被误分为 `early_budding`：`28` 个
- `spheroid` 被误分为 `cyst`：`18` 个

## 六、结果解释

本次结果说明：在人小肠类器官 crop 图像上，EfficientNet-B0 可以较好地区分四种形态亚型。

表现最好的类别：

- `cyst`
- `late_budding`

相对较难的类别：

- `early_budding`
- `spheroid`

主要错误仍然集中在：

- `cyst` 与 `early_budding` / `spheroid` 的混淆；
- `early_budding` 与 `late_budding` 的阶段边界混淆。

这些混淆符合类器官形态连续变化的特点。

## 七、输出文件

模型文件：

```text
models/human_intestine_b0_best.pth
models/human_intestine_b0_last.pth
```

结果文件：

```text
logs/human_intestine_b0/metrics.json
logs/human_intestine_b0/confusion_matrix.png
logs/human_intestine_b0/result_summary.md
```

## 八、当前结论

本次训练可以作为人小肠单数据域 expert baseline。

中文表述：

```text
人小肠类器官形态亚型四分类模型在验证集上取得了 89.07% 的准确率和 87.66% 的 macro-F1，说明单独在人小肠数据域内进行形态亚型分类具有较好的可行性。
```
