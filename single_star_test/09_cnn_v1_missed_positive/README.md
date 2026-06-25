# 09 CNN V1 Missed-Positive Specialization

## 目的

最后一次尝试：用最早 simple CNN，从 `sigma=2.5` 候选中找旧模型未召回的真星点作为正样本，再与负样本 1:1 混合，验证 simple CNN 是否能特化学会漏检真星。

## 架构

仍使用 simple CNN。实验脚本在 `code/scripts/train_missed_simple_cnn.py`。

## 状态

脚本已准备好，结果目录 `results/simple_cnn_missed_sigma2p5_seed42` 用于保存输出。
