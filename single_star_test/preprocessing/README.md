# preprocessing 目录说明

这个目录保存单帧星点提取项目的数据预处理、传统星点检测和早期 GAIA 评估脚本。它们不是模型训练脚本，主要负责把原始数据整理成 `single_star_test/data/data_model/`，以及在早期验证不同传统星点提取方法。

注意：部分脚本的默认参数仍保留整理前的旧路径，例如 `data/data_S30Pro`、`data/data_model`、`preprocessing/gaia_data`。在当前目录结构下，从仓库根目录运行时建议显式传入 `single_star_test/...` 路径。

## 数据处理主线

当前主线数据流是：

```text
single_star_test/data/data_S30Pro/
  原始 S30Pro 单帧 + 叠加图 + stack mask
        |
        | model_data_processing.py
        v
single_star_test/data/data_model/
  统一 image/mask/manifest 数据集
        |
        | candidate_scorer.build_dataset / V3 build_dataset
        v
single_star_test/data/candidate_scorer_*/
  CNN 候选 patch 训练数据
```

早期 GAIA 路线是辅助验证路线：

```text
datadownload.py
        |
        v
single_star_test/preprocessing/gaia_data/
        |
        | evaluate_mask_methods_gaia.py / evaluate_single_frames_gaia.py
        v
GAIA 对齐和传统星点检测评估 JSON
```

## 文件说明

### `model_data_processing.py`

主数据集生成脚本，也是当前最关键的 preprocessing 文件。

输入：

- `--root`：S30Pro 原始数据目录。整理后建议传 `single_star_test/data/data_S30Pro`。
- 每个观测组中的叠加图 `Stacked_*.fit`。
- 每个观测组对应的 `_sub` 单帧目录。

输出：

- `--output`：整理后的模型数据目录。整理后建议传 `single_star_test/data/data_model`。
- `train/images/`
- `train/masks/`
- `val/images/`
- `val/masks/`
- `stack_masks/`
- `manifest.csv`
- `summary.json`

处理逻辑：

1. 遍历 `data_S30Pro/` 的观测组。
2. 读取叠加图和对应单帧。
3. 由叠加图/stack mask 生成标签星点。
4. 用 WCS 和单帧候选点估计标签到单帧的平移/仿射对齐。
5. 将每张单帧写入标准 image/mask 目录。
6. 在 `manifest.csv` 中记录 `split_reason`，包括 `train`、`frame_holdout`、`coord_holdout`。

推荐运行方式：

```bash
python single_star_test/preprocessing/model_data_processing.py \
  --root single_star_test/data/data_S30Pro \
  --output single_star_test/data/data_model
```

如果目标目录已存在，需要确认不会覆盖重要数据后再使用 `--overwrite`。

### `mask_generator.py`

传统星点检测和 mask 生成工具库，同时也可以作为命令行脚本使用。

输入：

- 单个 FITS 文件，或一个 FITS 目录。
- 常见来源是 `single_star_test/data/data_S30Pro/` 中的单帧或叠加图。

输出：

- PNG mask，写到命令行指定的输出路径。
- 可选 sidecar JSON，保存 centroid 和调试信息。

支持的方法：

- `tetra3_like`
- `sextractor_like`
- `daofind_like`

它在项目里的作用：

- `model_data_processing.py` 调用它从叠加图/单帧中提取星点和 mask。
- GAIA 评估脚本调用它比较传统检测方法。
- 候选 scorer 路线里的 DAOFIND-like 思路也来自这里的早期实现。

示例：

```bash
python single_star_test/preprocessing/mask_generator.py \
  single_star_test/data/data_S30Pro \
  single_star_test/data_analysis/masks \
  --method daofind_like \
  --recursive \
  --write-json
```

### `compare_s30pro_centroids.py`

S30Pro 单帧与叠加图 centroid 数量对比脚本。

输入：

- `--root`：S30Pro 原始数据目录，整理后建议传 `single_star_test/data/data_S30Pro`。
- 每个观测组的叠加图目录和 `_sub` 单帧目录。

输出：

- 默认输出到终端，为 JSON 摘要。
- 如果需要保存，可以用 shell 重定向到 `single_star_test/data_analysis/`。

处理内容：

- 对每个叠加图提取星点数。
- 对该组单帧逐张提取星点数。
- 统计单帧星点数相对叠加图的比例、均值、中位数、最大最小值。

用途：

- 早期判断单帧质量是否足够。
- 发现坏帧、低质量组、FITS/JPG 差异。
- 辅助解释为什么单帧召回显著低于叠加图标签。

示例：

```bash
python single_star_test/preprocessing/compare_s30pro_centroids.py \
  --root single_star_test/data/data_S30Pro \
  > single_star_test/data_analysis/s30pro_centroid_compare.json
```

### `datadownload.py`

GAIA DR3 星表下载脚本。它属于早期 GAIA 标签/评估路线，不是当前 stack mask 主标签路线。

输入：

- RA/Dec 范围。
- 星等上限。
- 分块大小。

输出：

- 默认输出到脚本旁的 `gaia_data/`：
  - 分块 CSV：`single_star_test/preprocessing/gaia_data/*_chunks/`
  - 合并 CSV：`single_star_test/preprocessing/gaia_data/*.csv`
  - 配置 JSON：同名 `.json`

处理内容：

- 按天空区域分块查询 GAIA。
- 缓存每个 chunk。
- 合并成离线 CSV，供后续 GAIA 评估脚本读取。

示例：

```bash
python single_star_test/preprocessing/datadownload.py \
  --mag-limit 12 \
  --ra-min 0 --ra-max 360 \
  --dec-min -90 --dec-max 90 \
  --ra-step 20 --dec-step 20
```

### `evaluate_mask_methods_gaia.py`

叠加图层面的传统检测方法 GAIA 评估脚本。

输入：

- `--stack-root`：S30Pro 数据目录，整理后建议传 `single_star_test/data/data_S30Pro`。
- `--gaia-csv`：离线 GAIA catalog，整理后通常在 `single_star_test/preprocessing/gaia_data/`。
- 叠加图 `Stacked_*.fit`。

输出：

- `--out-json`：评估结果 JSON。整理后建议写到 `single_star_test/data_analysis/` 或 `single_star_test/preprocessing/gaia_data/`。

处理内容：

- 将 GAIA 星表通过 FITS WCS 投影到图像坐标。
- 用 `mask_generator.py` 的 `tetra3_like`、`sextractor_like`、`daofind_like` 提取星点。
- 与 GAIA 投影点匹配。
- 输出各方法的 completeness、limiting magnitude、分星等统计等。

示例：

```bash
python single_star_test/preprocessing/evaluate_mask_methods_gaia.py \
  --stack-root single_star_test/data/data_S30Pro \
  --gaia-csv single_star_test/preprocessing/gaia_data/gaia_g12p0_ra0p0_360p0_decm90p0_90p0_step20p0x20p0_pm.csv \
  --out-json single_star_test/data_analysis/gaia_extraction_eval.json
```

### `evaluate_single_frames_gaia.py`

单帧层面的传统检测方法 GAIA 评估脚本。

输入：

- `--root`：S30Pro 原始数据目录，整理后建议传 `single_star_test/data/data_S30Pro`。
- `--gaia-csv`：离线 GAIA catalog，整理后通常在 `single_star_test/preprocessing/gaia_data/`。
- 每个观测组中的叠加图和抽样单帧。

输出：

- `--out-json`：单帧 GAIA 评估结果 JSON。

处理内容：

- 从每个观测组抽取若干单帧。
- 用叠加图 WCS 和单帧 header 估计坐标变换。
- 可用传统检测点进一步估计平移修正。
- 将 GAIA 星点投影到单帧坐标。
- 比较 `tetra3_like`、`sextractor_like`、`daofind_like` 在单帧上的检测质量。

示例：

```bash
python single_star_test/preprocessing/evaluate_single_frames_gaia.py \
  --root single_star_test/data/data_S30Pro \
  --gaia-csv single_star_test/preprocessing/gaia_data/gaia_s30pro_4fields_g16_local.csv \
  --out-json single_star_test/data_analysis/gaia_single_frame_eval_g16.json
```

### `gaia_data/`

GAIA 下载和评估缓存目录。

内容可能包括：

- `*.csv`：合并后的 GAIA catalog。
- `*_chunks/`：分块下载缓存。
- `*.json`：下载配置或评估结果。

当前主训练标签来自 stack mask，不直接来自 GAIA，因此这个目录属于辅助数据，不应作为 CNN scorer 的标签来源。

## 已删除文件

### `tetra3_gaia_label.py`

旧的 tetra3 + GAIA 标签生成脚本。该路线后来不作为当前主训练标签来源，已按整理要求删除。当前标签以 `data_model/` 中的 stack mask/heatmap 标签为准。
