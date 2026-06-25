# 01 CNN V1 Simple Sigma=2.5

## 目的

低阈值 DAOFIND `sigma=2.5` 生成候选，裁剪候选中心 `31x31` patch，用最简单 CNN 二分类筛选候选点。

## 架构

Conv16/32/64 + BN/ReLU/Pool + MLP 输出 binary logit。原始训练脚本在 `code/train_candidate_scorer.py`。

## 结果

整图 spotcheck F1 约 `0.509`，明显优于默认 DAOFIND 的 `0.435`，但之后长期卡在 `0.50~0.51`。

## 结果文件

`results/candidate_scorer_sigma2p5*` 保存训练、resume 和 full/crop spotcheck 结果。
