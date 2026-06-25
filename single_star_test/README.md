# 单帧星点提取实验归档

这个目录只保留单帧星点提取相关的核心内容：

1. `data/`：主数据集和候选 patch 数据集。
2. `preprocessing/`：数据预处理、`data_model` 生成和传统星点检测工具。
3. `data_analysis/`：早期数据质量扫描、FITS/JPG 对比和 stack mask 调试输出。
4. `00_...` 到 `10_...`：按时间和路线拆分的主要训练或实验尝试。

模拟生成、论文资料、测试图片、tetra3 辅助库等非训练归档内容保留在目录外的 `../project_auxiliary/`。

## 实验目录

- `00_unet_heatmap/`：U-Net heatmap 分割路线。
- `01_cnn_v1_simple_sigma2p5/`：最早 simple CNN + DAOFIND `sigma=2.5`。
- `02_cnn_v1_sigma2_low_threshold/`：`sigma=2.0` 更低阈值候选实验。
- `03_cnn_v1_3ch/`：3 通道 patch 实验。
- `04_cnn_v1_hard_negative/`：hard negative mining 实验。
- `05_cnn_v1_numeric_unique_soft/`：numeric features、unique_soft、coord eval 等早期改进。
- `06_candidate_generation_probe/`：DAOFIND、LoG、adaptive LoG 候选生成上限分析。
- `07_cnn_v3_center_aware/`：V3 双尺度 center-aware scorer。
- `08_cnn_v3_recall_first/`：V3 recall-first 训练尝试。
- `09_cnn_v1_optimized_pipeline/`：用优化后的 shard/resume 数据管线重训 01 simple CNN。
- `10_cnn_v1_missed_positive/`：simple CNN missed-positive 特化训练。

## 当前结论

simple CNN 将默认 DAOFIND 的整图 F1 从约 `0.435` 提到约 `0.50~0.51`。后续 3ch、hard negative、numeric、V3、recall-first 均未稳定突破 `0.51`。V3 候选 oracle recall 约 `0.78~0.80`，但 scorer recall 仍只有约 `0.35`。
