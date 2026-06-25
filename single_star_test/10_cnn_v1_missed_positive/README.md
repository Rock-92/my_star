# 10 CNN V1 Missed-Positive Specialization

## 目的

最后一次尝试：基于 `09_eval_coord10` 的 10 张整图评估结果，找出 09 模型没有召回的标签星点，把这些漏检星点附近的 `sigma=2.5` 候选作为正样本，再采同数量远离标签的负样本，按 1:1 训练 simple CNN。

## 架构

仍使用 simple CNN。实验脚本在 `code/scripts/train_missed_simple_cnn.py`。

输入来源：

- `single_star_test/result_analysis/09_eval_coord10/report.json`
- `single_star_test/result_analysis/09_eval_coord10/per_sample.csv`
- `single_star_test/09_cnn_v1_optimized_pipeline/results/simple_cnn_v1_opt_sigma2p5_seed42/candidate_scorer_best.pt`

脚本会自动读取 09 在这 10 张图上的最佳阈值和 sample id。

## 状态

运行命令：

```bash
python -u single_star_test/10_cnn_v1_missed_positive/code/scripts/train_missed_simple_cnn.py \
  --data-root single_star_test/data/data_model \
  --config single_star_test/00_unet_heatmap/code/star_unet/config.json \
  --source-eval-dir single_star_test/result_analysis/09_eval_coord10 \
  --source-checkpoint single_star_test/09_cnn_v1_optimized_pipeline/results/simple_cnn_v1_opt_sigma2p5_seed42/candidate_scorer_best.pt \
  --out-dir single_star_test/10_cnn_v1_missed_positive/results/missed_from_09_coord10_seed42 \
  --candidate-methods daofind:2.5 \
  --epochs 10 \
  --batch-size 512 \
  --device cuda \
  --seed 42
```

输出：

- `candidate_scorer_best.pt`
- `candidate_scorer_last.pt`
- `summary.csv`
- `summary.json`
- `history.json`
- `metadata.json`

