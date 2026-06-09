# Candidate Scorer

This module trains a small CNN to rescore low-threshold DAOFind candidates.

The first experiment uses `daofind_like` with `sigma=2.5` to generate a high-recall candidate set, crops a small patch around each candidate, labels it by matching to the generated heatmap targets, then trains a binary scorer.

Build a starter dataset:

```powershell
python -m candidate_scorer.build_dataset --sigma 2.5 --out-dir data/candidate_scorer_sigma2p5
```

Train the scorer:

```powershell
python -m candidate_scorer.train --data-dir data/candidate_scorer_sigma2p5 --out-dir runs/candidate_scorer_sigma2p5
```
