# result_analysis 结果分析脚本

这个目录放独立的结果分析脚本和它们生成的结果。脚本与训练实验目录分开，避免把评估输出散落到各个模型目录。

## `evaluate_single_frame_methods.py`

用途：在同一批测试图上评估多种传统星点提取方式和最初 CNN V1。

默认评估对象：

- 数据：`single_star_test/data/data_model`
- split：`coord_holdout`
- 传统方法：
  - `daofind5=daofind:5.0`
  - `daofind2p5=daofind:2.5`
  - `log3p2=log:3.2`
  - `alog3p2=alog:3.2`
- CNN：01 simple CNN V1 checkpoint

输出：

- `single_frame_test_extractors/per_sample.csv`
- `single_frame_test_extractors/summary.csv`
- `single_frame_test_extractors/report.json`

`summary.csv` 中同时保存两类统计：

- `mean_precision`、`mean_recall`、`mean_f1`：先逐图计算，再对测试集图片做平均。
- `micro_precision`、`micro_recall`、`micro_f1`：把所有图片的 TP/FP/FN 汇总后再计算。

当前用户要求主要看 mean 指标。

## 云端运行命令

从仓库根目录执行：

```bash
source .venv/bin/activate

python -u single_star_test/result_analysis/evaluate_single_frame_methods.py \
  --data-root single_star_test/data/data_model \
  --config single_star_test/00_unet_heatmap/code/star_unet/config.json \
  --split-reason coord_holdout \
  --methods daofind5=daofind:5.0,daofind2p5=daofind:2.5,log3p2=log:3.2,alog3p2=alog:3.2 \
  --cnn-checkpoint single_star_test/01_cnn_v1_simple_sigma2p5/results/candidate_scorer_sigma2p5_full_resume60/candidate_scorer_best.pt \
  --cnn-candidate-methods daofind:2.5 \
  --cnn-thresholds 0.30:0.95:0.01 \
  --device cuda \
  --out-dir single_star_test/result_analysis/single_frame_test_extractors
```

如果云端没有 `candidate_scorer_best.pt`，可改用：

```bash
--cnn-checkpoint single_star_test/01_cnn_v1_simple_sigma2p5/results/candidate_scorer_sigma2p5_full_resume60/candidate_scorer.pt
```

先跑小样本验证命令：

```bash
python -u single_star_test/result_analysis/evaluate_single_frame_methods.py \
  --data-root single_star_test/data/data_model \
  --config single_star_test/00_unet_heatmap/code/star_unet/config.json \
  --split-reason coord_holdout \
  --count 2 \
  --methods daofind5=daofind:5.0,daofind2p5=daofind:2.5 \
  --cnn-thresholds 0.7,0.8 \
  --device cuda \
  --out-dir single_star_test/result_analysis/smoke
```
