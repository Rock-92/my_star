# 02 CNN V1 Sigma=2.0 Low Threshold

## 目的

进一步降低 DAOFIND 阈值到 `sigma=2.0`，提高候选 recall 上限。

## 结果

候选 recall 上升，但假阳性数量也大幅增加；CNN 筛选后整图 F1 仍约 `0.509`，没有突破。

## 结果文件

`results/candidate_scorer_sigma2_random*`。
