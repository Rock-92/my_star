# preprocessing 目录说明

这个目录保存单帧星点提取项目的数据预处理、传统星点检测和数据质量评估相关脚本。它们主要服务于 `single_star_test/data/data_model` 的生成和早期标签质量验证，不是模型训练脚本。

## 文件说明

### `model_data_processing.py`
主数据集生成脚本。负责把原始 S30Pro 单帧 FITS、叠加图和 stack mask 处理成训练用的 `data_model`：

- 生成 `train/images`、`train/masks`、`val/images`、`val/masks`
- 生成 `manifest.csv`
- 记录 `split_reason`：`train`、`frame_holdout`、`coord_holdout`
- 将叠加图/stack mask 标签对齐到单帧

这是当前 U-Net、CNN V1、CNN V3 数据来源中最关键的预处理脚本。

### `mask_generator.py`
传统星点检测和 mask 生成工具。包含背景扣除、噪声估计、连通域筛选和多种候选/掩膜生成方法：

- `daofind_like_mask`
- `sextractor_like_mask`
- `tetra3_like_mask`
- FITS/JPG 图像读取
- centroid 提取
- mask 写出

早期 DAOFIND/SExtractor-like 候选和一些数据质量分析依赖这里的函数。

### `datadownload.py`
GAIA 星表下载脚本。按天空区域分块查询、缓存、合并 GAIA catalog。属于早期星表标签路线的辅助脚本；当前主训练标签已经改为 stack mask，不再直接依赖它。

### `evaluate_mask_methods_gaia.py`
用 GAIA catalog 评估不同 FITS 星点 mask 方法的效果。用于早期比较传统检测方法的 completeness、limiting magnitude、匹配率等。

### `evaluate_single_frames_gaia.py`
评估单帧星点检测结果与 GAIA catalog 的匹配情况。包含单帧与叠加图/GAIA 投影之间的平移对齐估计逻辑。

### `compare_s30pro_centroids.py`
比较 S30Pro 单帧和叠加图中提取到的星点数量与 centroid 分布。用于早期数据质量扫描，帮助发现异常帧、低质量组和 FITS/JPG 差异。

### `gaia_data/`
GAIA 下载或缓存数据目录。属于外部星表辅助数据，不是当前模型直接训练数据。

### `README`
旧的极简说明文件，仅保留历史痕迹；主要说明以本文件 `README.md` 为准。

## 已删除文件

### `tetra3_gaia_label.py`
旧的 tetra3 + GAIA 标签生成脚本。该路线后来不作为当前主训练标签来源，已按整理要求删除。当前标签以 `data_model` 中 stack mask/heatmap 标签为准。
