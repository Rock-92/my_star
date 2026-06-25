# 数据目录说明

这个目录保存单帧星点提取实验用到的数据。请按“数据来源 -> 标注数据集 -> 候选 patch 数据集 -> 缓存”的顺序理解它，而不是按文件夹名字字母顺序理解。

核心标签来源是 `data_model/` 里的叠加图/stack mask 标签。DAOFIND/LoG 等传统方法只负责生成候选点，不作为最终真值标签。

## 1. 外部参考数据：`data_ZTF/`

`data_ZTF/` 是早期用于理解天文图像、星点分布和外部参考流程的数据目录。它不是当前 CNN scorer 的直接训练输入。

它的作用主要是：

- 作为早期星点检测、图像读取和可视化流程的参考数据。
- 帮助验证 FITS/JPG 读取、背景估计、星点候选提取等基础工具。
- 和当前 S30Pro 主线没有严格的一一对应关系。

当前 F1 评估、候选生成和训练主线都不依赖 `data_ZTF/`。

## 2. 原始观测数据：`data_S30Pro/`

`data_S30Pro/` 是当前项目的主要上游数据。它保存 S30Pro 拍摄得到的单帧图像、对应叠加图，以及 stack mask 生成所需的相关文件。

典型组织方式是按观测组存放：

- 叠加图目录：包含 `Stacked_*.fit` 等叠加结果。
- 单帧目录：通常是同名目录加 `_sub`，包含该组单帧 `*.fit`。
- 部分组可能同时包含 JPG、预览图或中间文件。

这份数据本身还不是训练数据。它需要经过 `single_star_test/preprocessing/model_data_processing.py` 处理，才能变成模型统一读取的 `data_model/`。

## 3. 主模型数据：`data_model/`

`data_model/` 是由 `data_S30Pro/` 生成的正式训练/评估数据集，也是后续 U-Net、CNN V1、CNN V3 的共同标签来源。

生成脚本：

```bash
python single_star_test/preprocessing/model_data_processing.py \
  --root single_star_test/data/data_S30Pro \
  --output single_star_test/data/data_model
```

生成逻辑：

1. 遍历 `data_S30Pro/` 中每个观测组。
2. 找到该组叠加图 `Stacked_*.fit` 和对应 `_sub` 单帧。
3. 从叠加图/stack mask 提取星点标签。
4. 将叠加图标签对齐到每张单帧。
5. 输出标准化后的 image/mask 文件和 `manifest.csv`。

典型结构：

- `train/images/`：训练物理目录下的单帧图像。
- `train/masks/`：对应 heatmap/mask 标签。
- `val/images/`：验证物理目录下的单帧图像。
- `val/masks/`：对应 heatmap/mask 标签。
- `stack_masks/`：从叠加图生成或保存的 stack-level mask。
- `manifest.csv`：每张单帧的路径、组号、标签对齐信息和真实 split。
- `summary.json`：生成统计。

重要：历史脚本中有 `train/val` 物理目录，但真正实验划分以 `manifest.csv` 的 `split_reason` 为准：

- `train`：646 张，用于训练。
- `frame_holdout`：58 张，用于开发集评估、选模型、选阈值。
- `coord_holdout`：66 张，最终测试集；方案定型前不参与调参。

## 4. 早期候选 patch 数据：`candidate_scorer_sigma2p5_full/`

这是 simple CNN V1 的主要训练数据，来自 `data_model/`。

生成目的：验证“低阈值 DAOFIND 候选 + 31x31 patch CNN 打分”是否能超过默认 DAOFIND。

关键配置：

- 候选生成：DAOFIND-like，`sigma=2.5`
- crop：固定/中心 `1024x1024`
- patch：`31x31`，单通道
- 正样本：候选点距离 stack 标签星点 `<= 4 px`
- ignore：距离 `4~6 px`
- 负样本：距离 `>= 6 px`
- 每图最多负样本：约 `800`
- train：`625758` patches，正样本 `108958`，负样本 `516800`
- val：`244013` patches，正样本 `21752`，负样本 `222261`

近似生成命令：

```bash
python -m candidate_scorer.build_dataset \
  --sigma 2.5 \
  --crop-size 1024 \
  --patch-size 31 \
  --max-negatives-per-image 800 \
  --out-dir single_star_test/data/candidate_scorer_sigma2p5_full
```

实验结论：整图 spotcheck F1 从默认 DAOFIND 的约 `0.435` 提升到约 `0.50~0.51`，是目前最有效的简单基线。

## 5. 更低阈值候选冒烟数据：`candidate_scorer_sigma2_random_smoke/`

这是 `sigma=2.0` 路线的小样本 smoke 数据，来自 `data_model/`。

生成目的：先确认更低阈值、随机 crop 和训练读取流程能跑通，再决定是否生成大数据。

关键配置：

- 候选生成：DAOFIND-like，`sigma=2.0`
- crop：`512x512`
- patch：`31x31`，单通道
- train：`3409` patches，正样本 `209`，负样本 `3200`
- val：`2297` patches，正样本 `87`，负样本 `2210`

## 6. 更低阈值候选全量数据：`candidate_scorer_sigma2_random2/`

这是 `sigma=2.0` 的全量 random crop 数据，来自 `data_model/`。

生成目的：提高候选召回上限，观察 scorer 能否从更多候选里找回漏检真星点。

关键配置：

- 候选生成：DAOFIND-like，`sigma=2.0`
- crop：random `1024x1024`
- 每张图：`2` 个随机 crop
- patch：`31x31`，单通道
- 正样本半径：`4 px`
- ignore 半径：`6 px`
- train：`1285989` patches，正样本 `252389`，负样本 `1033600`
- val：`1184236` patches，正样本 `50449`，负样本 `1133787`

近似生成命令：

```bash
python -m candidate_scorer.build_dataset \
  --sigma 2.0 \
  --channels 1 \
  --crop-size 1024 \
  --crop-mode random \
  --crops-per-image 2 \
  --max-negatives-per-image 800 \
  --out-dir single_star_test/data/candidate_scorer_sigma2_random2
```

实验结论：候选 oracle recall 明显提高，但候选假阳性太多，CNN 筛选后整图 F1 仍约 `0.50~0.51`。

## 7. 数值特征冒烟数据：`candidate_scorer_sigma2p5_num_smoke/`

这是 `sigma=2.5 + numeric features` 的小样本 smoke 数据，来自 `data_model/`。

生成目的：验证候选数值特征保存、归一化和训练读取逻辑。

关键配置：

- 候选生成：DAOFIND-like，`sigma=2.5`
- crop：random `512x512`
- 每张图：`1` 个 crop
- patch：`31x31`，单通道
- numeric features：10 维，包括中心亮度、背景残差、matched response、局部均值/方差、SNR、归一化坐标、边缘距离
- train：`1012` patches，正样本 `69`，负样本 `943`
- val：`476` patches，正样本 `42`，负样本 `434`

## 8. unique_soft 计划数据：`candidate_scorer_sigma2p5_num_unique_soft_random2/`

这是计划中的 `sigma=2.5 + numeric features + unique_soft + random crop x2` 数据目录。

计划生成目的：减少顺序相关的伪标签噪声，让每个标签星点最多只分配一个正候选，其余近邻候选设为 ignore。

计划配置：

- 候选生成：DAOFIND-like，`sigma=2.5`
- crop：random `1024x1024`
- 每张图：`2` 个 crop
- patch：`31x31`，单通道
- numeric features：10 维
- label mode：`unique_soft`
- soft label sigma：`2.0 px`
- max negatives per image：`800`

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
  --out-dir single_star_test/data/candidate_scorer_sigma2p5_num_unique_soft_random2
```

注意：当前本地目录为空，占位保留是为了记录这条路线；完整数据没有归档到本机。

## 9. V2 shard/resume 冒烟数据：`candidate_scorer_v2_smoke_20260613/`

这是新版数据构建管线的 smoke 数据，来自 `data_model/`。

生成目的：验证 shard/resume、配置指纹、双尺度输入和候选元数据保存。

关键配置：

- schema version：2
- candidate methods：`daofind:2.5`
- 候选去重半径：`2.5 px`
- patch：中心 `31x31` + 上下文 `63x63`
- 输入通道：6
- numeric features：16 维
- train split reason：`train`
- val split reason：`frame_holdout`
- shard：`shards/train_000000.npz`、`shards/val_000000.npz`
- 样本量：train 1 张、val 1 张

## 10. V3 center-aware 冒烟数据：`candidate_scorer_v3_smoke_20260613/`

这是 V3 center-aware scorer 的 smoke 数据，来自 `data_model/`。

生成目的：验证 V3 多任务模型需要的数据结构，包括三分类标签、中心先验通道、质量分支和 offset 回归。

关键配置：

- schema version：3
- candidate methods：`daofind:2.5`
- 候选去重半径：`2.5 px`
- patch：中心 `31x31` + 上下文 `63x63`
- 输入通道：6
- numeric features：16 维，包括候选响应、局部噪声分位数、形状矩、多尺度/来源统计等
- train split reason：`train`
- val split reason：`frame_holdout`
- shard：`shards/train_000000.npz`、`shards/val_000000.npz`
- 样本量：train 1 张、val 1 张

## 11. 候选缓存：`candidate_cache/`

`candidate_cache/` 是整图评估或候选探针过程中产生的缓存。

内容：

- 每个哈希子目录对应一组候选生成配置。
- 子目录里按样本保存 `.npz`，例如 `sample_000011.npz`。

生成目的：缓存候选点、标签点和部分图像特征，避免反复跑整图 DAOFIND/LoG。缓存可以删除，后续评估会按配置重建。

## 使用建议

- 想理解数据源头：先看 `data_ZTF/` 和 `data_S30Pro/`。
- 想复现主训练数据：看 `data_S30Pro/` -> `preprocessing/model_data_processing.py` -> `data_model/`。
- 想复现 simple CNN V1：看 `candidate_scorer_sigma2p5_full/`。
- 想理解为什么 `sigma=2.0` 没突破：看 `candidate_scorer_sigma2_random2/`。
- 想看 V3 数据结构：看 `candidate_scorer_v3_smoke_20260613/`，但它只是 smoke 数据，不是全量训练集。
