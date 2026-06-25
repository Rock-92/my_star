# 数据目录总览

这个目录保存单帧星点提取实验用到的正式数据、中间候选数据和少量缓存。核心标签来源是 `data_model` 里的叠加图/stack mask 生成标签，不是单帧 DAOFIND 自标注。

## 数据划分原则

`data_model/manifest.csv` 中通过 `split_reason` 区分用途：

- `train`：646 张，用于训练候选 scorer 或密集模型。
- `frame_holdout`：58 张，用于开发集评估、选模型、选全局阈值。
- `coord_holdout`：66 张，最终测试集；方案定型前不参与调参。

## 子目录说明

### `data_model/`
主数据集，约 770 张样本。由 `single_star_test/preprocessing/model_data_processing.py` 从原始 S30Pro 单帧、叠加图和 stack mask 生成。

用途：

- U-Net heatmap 路线的 image/mask 数据源。
- CNN 候选 scorer 的真实星点标签来源。
- `candidate_scorer.evaluate` 和 `candidate_scorer.probe_candidates` 的整图评估来源。

典型结构：

- `train/images`、`train/masks`
- `val/images`、`val/masks`
- `manifest.csv`

注意：历史脚本里 `train/val` 是物理目录名，真正实验划分以 `manifest.csv` 的 `split_reason` 为准。

### `candidate_scorer_sigma2p5_full/`
早期 simple CNN V1 使用的主要候选 patch 数据集。

关键配置：

- 候选生成：DAOFIND-like，`sigma=2.5`
- crop：中心/固定 `1024x1024`
- patch：`31x31`，单通道
- 正样本半径：`4 px`
- ignore 半径：`6 px`
- 每图最多保留约 `800` 个负样本
- train：`625758` patches，其中正样本 `108958`，负样本 `516800`
- val：`244013` patches，其中正样本 `21752`，负样本 `222261`

生成目的：验证“低阈值候选 + simple CNN 打分”是否优于默认 DAOFIND。结果证明整图 F1 从约 `0.435` 提升到约 `0.50~0.51`，是后续路线的起点。

近似生成命令：

```bash
python -m candidate_scorer.build_dataset \
  --sigma 2.5 \
  --crop-size 1024 \
  --patch-size 31 \
  --max-negatives-per-image 800 \
  --out-dir data/candidate_scorer_sigma2p5_full
```

### `candidate_scorer_sigma2_random_smoke/`
`sigma=2.0` 随机 crop 小样本冒烟数据。

关键配置：

- 候选生成：DAOFIND-like，`sigma=2.0`
- crop：`512x512`
- patch：`31x31`，单通道
- train：`3409` patches，正样本 `209`，负样本 `3200`
- val：`2297` patches，正样本 `87`，负样本 `2210`

生成目的：先用很小数据确认 `sigma=2.0`、random crop 和训练管线能跑通，再决定是否生成全量。

### `candidate_scorer_sigma2_random2/`
`sigma=2.0` 全量 random crop 数据集。

关键配置：

- 候选生成：DAOFIND-like，`sigma=2.0`
- crop：random `1024x1024`
- 每张图 `2` 个随机 crop
- patch：`31x31`，单通道
- 正样本半径：`4 px`
- ignore 半径：`6 px`
- train：`1285989` patches，其中正样本 `252389`，负样本 `1033600`
- val：`1184236` patches，其中正样本 `50449`，负样本 `1133787`

生成目的：提高候选召回上限。实验结论是 raw candidate recall 上升，但假阳性数量过大，CNN 筛选后整图 F1 仍约 `0.50~0.51`，没有突破。

近似生成命令：

```bash
python -m candidate_scorer.build_dataset \
  --sigma 2.0 \
  --channels 1 \
  --crop-size 1024 \
  --crop-mode random \
  --crops-per-image 2 \
  --max-negatives-per-image 800 \
  --out-dir data/candidate_scorer_sigma2_random2
```

### `candidate_scorer_sigma2p5_num_smoke/`
`sigma=2.5 + numeric features` 小样本冒烟数据。

关键配置：

- 候选生成：DAOFIND-like，`sigma=2.5`
- crop：random `512x512`
- 每张图 `1` 个 crop
- patch：`31x31`，单通道
- numeric features：10 维，包括中心亮度、背景残差、matched response、局部均值/方差、SNR、归一化坐标、边缘距离等
- train：`1012` patches，正样本 `69`，负样本 `943`
- val：`476` patches，正样本 `42`，负样本 `434`

生成目的：验证数值特征融合的数据保存、归一化和训练读取逻辑。

### `candidate_scorer_sigma2p5_num_unique_soft_random2/`
计划中的 `sigma=2.5 + numeric features + unique_soft + random crop x2` 数据目录。

计划配置：

- 候选生成：DAOFIND-like，`sigma=2.5`
- crop：random `1024x1024`
- 每张图 `2` 个 crop
- patch：`31x31`，单通道
- numeric features：10 维
- 标签模式：`unique_soft`
- soft label sigma：`2.0 px`
- 每个标签星点最多分配一个候选为正样本，其余近邻候选 ignore

注意：当前本地目录几乎为空，仅保留占位 README，说明这份全量数据没有完整归档到本机。

对应命令：

```bash
python -m candidate_scorer.build_dataset \
  --sigma 2.5 \
  --channels 1 \
  --train-samples 0 \
  --val-samples 0 \
  --crop-size 1024 \
  --crop-mode random \
  --crops-per-image 2 \
  --max-negatives-per-image 800 \
  --label-mode unique_soft \
  --soft-label-sigma-px 2.0 \
  --out-dir data/candidate_scorer_sigma2p5_num_unique_soft_random2
```

### `candidate_scorer_v2_smoke_20260613/`
V2/V3 数据管线冒烟数据，采用 shard/resume 格式。

关键配置：

- schema version：2
- candidate methods：`daofind:2.5`
- 候选去重半径：`2.5 px`
- patch：中心 `31x31` + 上下文 `63x63`
- 输入通道：6
- numeric features：16 维
- split：train 使用 `train`，val 使用 `frame_holdout`
- shard：`shards/train_000000.npz`、`shards/val_000000.npz`
- 样本量：train 1 张、val 1 张

生成目的：验证新版分片写入、可续跑、双尺度输入、numeric features 和三分类标签结构是否能跑通。

### `candidate_scorer_v3_smoke_20260613/`
V3 center-aware scorer 的冒烟数据。

关键配置：

- schema version：3
- candidate methods：`daofind:2.5`
- 候选去重半径：`2.5 px`
- patch：中心 `31x31` + 上下文 `63x63`
- 输入通道：6，包括归一化原图、背景/滤波响应以及中心先验/坐标类通道
- numeric features：16 维，包括候选响应、局部噪声分位数、形状矩、多尺度/来源统计等
- split：train 使用 `train`，val 使用 `frame_holdout`
- shard：`shards/train_000000.npz`、`shards/val_000000.npz`
- 样本量：train 1 张、val 1 张

生成目的：验证 V3 多任务 scorer 所需的数据结构：三分类、质量分支、offset 回归、score-NMS 前所需的候选元数据。

### `candidate_cache/`
候选评估缓存目录。

内容：

- 每个哈希子目录对应一组候选生成配置。
- 里面按样本保存 `.npz`，例如 `sample_000011.npz`。

生成目的：缓存候选点、标签点和部分图像特征，避免反复跑整图 DAOFIND/LoG 候选生成。缓存可删除，后续评估会按配置重建。

### `data_S30Pro/`
原始或外部整理的 S30Pro 数据目录。

用途：

- `data_model` 的上游来源之一。
- 包含单帧 FITS/JPG、叠加图或相关中间文件。
- 不直接作为 CNN scorer 的训练输入，通常先经过 `preprocessing/model_data_processing.py` 生成 `data_model`。

### `data_ZTF/`
ZTF 相关原始或参考数据目录。

用途：

- 早期对齐、星点质量分析或外部参考实验。
- 当前单帧 scorer 主线不直接依赖它。

## 读取建议

- 训练 CNN V1：优先看 `candidate_scorer_sigma2p5_full/`、`candidate_scorer_sigma2_random2/`。
- 训练 V3：正式全量数据如果存在应使用 shard/resume 格式；本目录里的 `candidate_scorer_v3_smoke_20260613/` 只是冒烟数据。
- 做整图评估：使用 `data_model/` 和 `candidate_cache/`，不要用 crop F1 替代整图 F1。
