# 09 CNN V1 Optimized Pipeline Retrain

## 目的

这次实验是把最早 `01_cnn_v1_simple_sigma2p5` 的 simple CNN 架构拿回来，但不再使用最早的临时数据生成方式，而是使用后期优化过的数据管线重新生成训练数据后再训练。

核心问题：

> 如果模型架构退回最简单的 31x31 单通道 CNN，但数据划分、标签匹配、shard/resume、分层 crop 和候选元数据都使用新版管线，能不能比早期 simple CNN 更稳？

## 和 01 的区别

模型仍然是 01 的 simple CNN：

- 输入：`31x31` 单通道 patch
- 结构：Conv-BN-ReLU-Pool x3 + MLP 二分类头
- 输出：候选点是否是真实中心星点

但数据管线换成优化版：

- 从 `data_model/manifest.csv` 的 `split_reason` 选择 train 和 `frame_holdout`
- 支持 shard/resume
- 支持 `candidate-methods`
- 支持 stratified crop，覆盖中心、边缘和四角
- 使用新版三分类标签，其中 class 2 为真实中心，class 0/1 在本实验中都作为负样本

## 数据生成

推荐先在云端生成全量数据。注意这里生成的是 V3 schema shard 数据，但训练时只取中心 `31x31` 的第一个图像通道，因此模型仍是 simple CNN。

Linux/云端命令示例：

```bash
export PYTHONPATH="$PWD/single_star_test/07_cnn_v3_center_aware/code:$PWD/single_star_test/00_unet_heatmap/code:$PWD/single_star_test"
DATA_ROOT="$PWD/single_star_test/data/data_model"
CONFIG="$PWD/single_star_test/00_unet_heatmap/code/star_unet/config.json"
OUT_DIR="$PWD/single_star_test/data/candidate_scorer_v1_opt_sigma2p5_stratified"

python -u -m candidate_scorer.build_dataset \
  --data-root "$DATA_ROOT" \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --candidate-methods daofind:2.5 \
  --dedup-radius-px 2.5 \
  --crop-size 1024 \
  --crop-mode stratified \
  --crops-per-image 5 \
  --patch-size 31 \
  --context-patch-size 63 \
  --train-split-reason train \
  --val-split-reason frame_holdout \
  --train-samples 0 \
  --val-samples 0 \
  --max-negatives-per-crop 300 \
  --max-offcenter-per-crop 100 \
  --shard-size 50000 \
  --resume \
  --seed 42
```

如果资源紧张，可以先把 `--train-samples 80 --val-samples 12` 当 smoke test。

## 训练

训练脚本：

```text
code/scripts/train_simple_from_v3_shards.py
```

全量训练命令：

```bash
DATA_DIR="$PWD/single_star_test/data/candidate_scorer_v1_opt_sigma2p5_stratified"
OUT_DIR="$PWD/single_star_test/09_cnn_v1_optimized_pipeline/results/simple_cnn_v1_opt_sigma2p5_seed42"

python -u single_star_test/09_cnn_v1_optimized_pipeline/code/scripts/train_simple_from_v3_shards.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --epochs 30 \
  --batch-size 512 \
  --pos-neg-ratio 1 \
  --device cuda \
  --seed 42
```

脚本会保存：

- `candidate_scorer_best.pt`
- `candidate_scorer_last.pt`
- `history.csv`
- `summary.json`

## 解释

这个实验的验证价值在于隔离变量：

- 如果结果明显好于 01，说明早期 simple CNN 没吃到好数据，数据管线仍有价值。
- 如果仍卡在约 `0.50~0.51`，说明瓶颈更可能在“patch scorer 本身无法稳定区分低 SNR 真星和背景假阳性”，继续堆候选 scorer 的收益会很低。

注意：当前训练脚本保存的是 patch 级验证 F1，不等价于完整整图 micro-F1。最终仍需要另写或复用 legacy full-frame evaluator，在固定 `frame_holdout` 上评估。
