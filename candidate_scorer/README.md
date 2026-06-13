# Candidate Scorer V2

The v2 pipeline uses mixed high-recall candidates, deterministic one-to-one labels,
dual-scale center-aware patches, numeric features, coordinate regression, and score NMS.
Model selection uses full `frame_holdout` frames. Keep `coord_holdout` sealed until the
configuration and global score threshold are final.

## 1. Candidate ceiling

```powershell
python -m candidate_scorer.probe_candidates `
  --split-reason frame_holdout `
  --out-dir runs/candidate_probe_dev
```

Choose the smallest candidate set with recall at least 0.80.

## 2. Resumable dataset

```powershell
python -m candidate_scorer.build_dataset `
  --candidate-methods "daofind:2.0,daofind:2.5,sextractor:1.5" `
  --crop-mode stratified `
  --crops-per-image 5 `
  --train-split-reason train `
  --val-split-reason frame_holdout `
  --out-dir data/candidate_scorer_v2
```

Restart the same command with `--resume` after interruption. A changed configuration
is rejected by its fingerprint.

## 3. Train and mine hard negatives

```powershell
python -m candidate_scorer.train `
  --data-dir data/candidate_scorer_v2 `
  --out-dir runs/candidate_scorer_v2_seed42 `
  --epochs 30 --amp --seed 42

python -m candidate_scorer.build_hard_dataset `
  --checkpoint runs/candidate_scorer_v2_seed42/candidate_scorer_best.pt `
  --out-dir data/candidate_scorer_v2_hard

python -m candidate_scorer.train `
  --data-dir data/candidate_scorer_v2 `
  --hard-negative-data-dir data/candidate_scorer_v2_hard `
  --out-dir runs/candidate_scorer_v2_hard_seed42 `
  --epochs 20 --amp --seed 42
```

## 4. Final locked evaluation

Select the checkpoint ensemble and threshold only from `frame_holdout`, then run
`coord_holdout` once:

```powershell
python -m candidate_scorer.evaluate `
  --checkpoints "runs/seed42/candidate_scorer_best.pt,runs/seed43/candidate_scorer_best.pt,runs/seed44/candidate_scorer_best.pt" `
  --split-reason coord_holdout `
  --fixed-threshold DEV_SELECTED_THRESHOLD `
  --out-dir runs/candidate_scorer_final_test
```
