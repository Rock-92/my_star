# 数据目录总览

这里存放单帧星点提取实验使用过的数据和中间数据集。每个正式子目录内都有 `README.md`，说明生成方式、用途和划分信息。

核心目录：

- `data_model/`：主数据集，770 张，`train=646`，`frame_holdout=58`，`coord_holdout=66`。
- `candidate_scorer_sigma2*` / `candidate_scorer_sigma2p5*`：早期 simple CNN scorer 的候选 patch 数据。
- `candidate_scorer_v2_smoke_20260613/`、`candidate_scorer_v3_smoke_20260613/`：V2/V3 数据管线 smoke 数据。
- `candidate_cache/`：评估候选缓存，可重建。
- `data_S30Pro/`、`data_ZTF/`：原始或外部数据目录。
- `tmphh6pz7tm/`：临时目录，当前不作为正式训练数据记录。
