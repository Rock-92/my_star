# 07 CNN V3 Center-Aware Scorer

## 目的

重建数据管线和 scorer：shard/resume、双尺度 patch、中心 Gaussian/坐标通道、numeric features、三分类 + quality + offset。

## 架构

代码在 `code/candidate_scorer/`。核心模型是双尺度 ResNet-ish `CenterAwareScorer`。

## 结果

58 张 full `frame_holdout`：`micro-F1≈0.493`，`precision≈0.826`，`recall≈0.351`，candidate oracle recall 约 `0.782`。

## 结果文件

`results/scorers/candidate_scorer_log3p2_seed42`，以及 `results/runs_local/` 中的本地副本。
