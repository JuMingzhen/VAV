# VAV Course Project Report Outline

## 1. Research Question
- 我们的问题：ViT 在细粒度鸟类分类训练过程中，attention 是否会逐渐聚焦到更有判别力的局部区域？
- 课程连接：把这个过程和认知科学中的 selective attention、global-to-local processing、feature integration 联系起来讨论。

## 2. Motivation
- 实际应用价值：细粒度物种识别可以用于生态监测、野外观测、自动化分类辅助。
- 课程价值：不仅关心准确率，也关心模型“看哪里”、这些注意模式怎样变化，以及这些变化能否用认知概念描述。

## 3. Theoretical Foundations
- Dosovitskiy et al. (2021): ViT 的基础架构。
- Abnar and Zuidema (2020): attention rollout。
- Chefer et al. (2021): attention 不是完整解释，需要谨慎表述。
- Raghu et al. (2021): ViT 与 CNN 的表征差异。
- Navon (1981): global precedence。
- Treisman and Gelade (1980): Feature Integration Theory。

## 4. Experimental Setup
- 数据集：CUB-200-2011。
- 模型：预训练 `google/vit-base-patch16-224`，在 CUB 上微调。
- 保存节点：`epoch_000`, `epoch_001`, `epoch_003`, `epoch_005`, `final`。
- 分析对象：
  - rollout heatmap
  - all-layers average heatmap
  - CLS attention at selected layers
- 指标：
  - validation accuracy
  - loss
  - attention entropy
  - top-k patch concentration
  - bbox overlap

## 5. Results to Include
- 训练曲线图。
- attention 指标随 epoch 的变化曲线。
- 同一图片跨 checkpoint 的 attention overlay。
- 成功案例与失败案例。

## 6. Discussion
- attention 可以作为“模型信息选择方式”的线索。
- 训练后 attention 更集中，不代表模型真的拥有人的注意机制，只能说它表现出可比较的模式。
- 结合课程概念讨论“相似”和“边界”，避免过度类比。

## 7. Limitations
- attention 不是完整解释。
- bbox overlap 只能粗略衡量是否聚焦关键区域。
- 细粒度分类结果仍受数据偏差和背景线索影响。
