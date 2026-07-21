# 肠道类器官形态亚型分类错误分析

基于四组实验的综合分析：`intestine_subtype_b0` / `human_intestine_b0` / `mouse_intestine_b0` / `intestine_subtype_b0_no_sampler`。

## 一、跨实验对比总览

| 实验 | Acc | Bal Acc | Macro-F1 | cyst F1 | early F1 | late F1 | spheroid F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 人+小鼠 (weighted) | 92.27 | 92.18 | 90.99 | 94.62 | 88.76 | 92.58 | 87.99 |
| 人+小鼠 (no sampler) | 92.51 | 91.77 | 91.02 | 95.39 | 88.12 | 91.52 | 89.04 |
| 仅人小肠 | 89.07 | 89.58 | 87.66 | 92.64 | 83.98 | 87.73 | 86.31 |
| 仅小鼠小肠 | 88.66 | 89.16 | 86.67 | 92.30 | 83.46 | 88.97 | 81.93 |

**结论 1**：联合训练（人+小鼠）在所有指标上均优于单域训练，验证了跨物种小肠类器官形态特征可共享。

## 二、四种系统性错误模式

### 2.1 cyst ↔ spheroid（囊状 ↔ 球体）

| 实验 | cyst→spheroid | % of cyst | spheroid→cyst | % of spheroid |
| --- | ---: | ---: | ---: | ---: |
| 人+小鼠 (weighted) | 70 | 3.0% | 37 | 8.6% |
| 人+小鼠 (no sampler) | 60 | 2.5% | 33 | 7.7% |
| 仅人 | 40 | 3.8% | 18 | 8.8% |
| 仅小鼠 | 68 | 5.3% | 20 | 8.9% |

**生物学原因**：cyst 和 spheroid 在外观上都偏圆形。cyst 是 lumen-forming 结构，spheroid 是密实的球体。在 224×224 低分辨率裁剪图中，两者的视觉差异可能很微弱。spheroid→cyst 混淆率（~8%）远高于 cyst→spheroid（~3%），说明模型倾向于把模糊的 spheroid 判为 cyst——这是 cyst 样本量远大于 spheroid（2357 vs 429）导致的类别偏差。

**跨域差异**：小鼠小肠的 cyst→spheroid 混淆（5.3%）高于人小肠（3.8%），提示小鼠 spheroid 可能更接近 cyst 外观。

### 2.2 early_budding ↔ late_budding（早出芽 ↔ 晚出芽）

| 实验 | early→late | % of early | late→early | % of late |
| --- | ---: | ---: | ---: | ---: |
| 人+小鼠 (weighted) | 61 | 5.8% | 41 | 5.9% |
| 人+小鼠 (no sampler) | 75 | 7.2% | 44 | 6.3% |
| 仅人 | 46 | 9.3% | 28 | 9.5% |
| 仅小鼠 | 50 | 9.1% | 37 | 9.2% |

**生物学原因**：这是四种模式中最直观的一个。类器官从 early budding 到 late budding 是一个连续过程，中间状态必然存在。混淆率在单域模型上约 9%，联合训练后降到 ~6%，说明多样化样本帮助模型学习更明确的决策边界。

### 2.3 cyst → early_budding（囊状 → 早出芽）

| 实验 | cyst→early_budding | % of cyst |
| --- | ---: | ---: |
| 人+小鼠 (weighted) | 103 | 4.4% |
| 人+小鼠 (no sampler) | 59 | 2.5% |
| 仅人 | 77 | 7.3% |
| 仅小鼠 | 80 | 6.2% |

这是 weighted sampler 副作用最明显的模式：启用 weighted sampler 时混淆率（4.4%）比不启用（2.5%）高很多。weighted sampler 过采样 minority 类（early/late/spheroid），可能让模型对 cyst 的边界判断更激进，把带有轻微出芽迹象的 cyst 错判为 early_budding。

### 2.4 spheroid → cyst（球体 → 囊状）

已经在上文 2.1 中合并分析。

## 三、跨域差异

| 现象 | 仅人 | 仅小鼠 | 联合 |
| --- | ---: | ---: | ---: |
| spheroid F1 | 86.31 | 81.93 | 87.99 |
| cyst→early_budding | 7.3% | 6.2% | 4.4% |
| cyst→spheroid | 3.8% | 5.3% | 3.0% |

- **小鼠 spheroid 明显更难**（F1 低 4.4 个百分点）。小鼠小肠 crop 来自 YOLO bbox，可能裁剪精度不如人小肠的预裁剪（224×224 标准 crop），边缘信息丢失。
- **仅人小肠的 cyst→early_budding 混淆更严重**（7.3% vs 6.2%），但联合训练后显著降低（4.4%），说明小鼠数据提供了额外的 early budding 样例。

## 四、Weighted Sampler 效应

weighted 和 no-sampler 的 overall 指标几乎一样（macro-F1: 90.99 vs 91.02），但 per-class 分布不同：

| 类别 | weighted F1 | no-sampler F1 | 差异 |
| --- | ---: | ---: | ---: |
| cyst | 94.62 | 95.39 | -0.77 |
| early_budding | 88.76 | 88.12 | +0.64 |
| late_budding | 92.58 | 91.52 | +1.06 |
| spheroid | 87.99 | 89.04 | -1.05 |

weighted sampler 并没有像预期的那样帮助 minority 类——spheroid 反而变差了。原因可能是：当前类别不均衡程度（5.5:1）在 label smoothing + 预训练 backbone 的条件下并不严重，强行 reweighting 反而引入噪声。

**建议**：当前数据规模下，不用 weighted sampler 即可。

## 五、下一步

1. **可视化困难样本**：从验证集中随机抽取 spheroid→cyst 和 early→late 的 misclassified 样本，人工判读标记是否有争议
2. **标注质量审计**：如果大量"误分类"实际上是标注不准确，需要在 report 中讨论 inter-rater variability
3. **spheroid 增强**：考虑对 spheroid 类增加数据增强（尤其是小鼠），或收集更多 spheroid 样本
4. **early/late budding 边界定义**：如果未来有生物学协作，可以考虑按出芽数量做定量分级（<3 buds = early, ≥3 buds = late），而不是依赖视觉判断

## 六、误分类样本导出流程

已新增脚本：

```text
export_misclassified_samples.py
```

用途：

- 读取 `metadata/training_manifest.csv` 的验证集样本
- 加载指定 intestinal subtype classifier checkpoint
- 导出逐样本预测表 `predictions.csv`
- 导出误分类样本表 `misclassified.csv`
- 为四类重点错误生成图片面板：
  - `early_budding -> late_budding`
  - `late_budding -> early_budding`
  - `spheroid -> cyst`
  - `cyst -> spheroid`

推荐命令：

```bash
python export_misclassified_samples.py \
  --checkpoint models/intestine_subtype_b0_best.pth \
  --run-name intestine_subtype_b0 \
  --output-dir logs/intestine_subtype_b0/error_examples \
  --max-per-pair 20
```

当前注意事项：

- 该流程需要 `models/intestine_subtype_b0_best.pth` 或等价的主 baseline checkpoint。
- 当前仓库已有 `logs/intestine_subtype_b0/metrics.json` 和混淆矩阵，但本地 `models/` 目录未发现 `intestine_subtype_b0_best.pth`，因此需要恢复该 checkpoint 或重新训练 baseline 后再运行导出。
