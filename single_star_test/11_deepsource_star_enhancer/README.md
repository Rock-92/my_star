# 11 DeepSource Star Enhancer

## 目的

复现 DeepSource 的核心思想：不再让 CNN 对候选 patch 做真假二分类，而是让一个很小的全卷积网络把单帧图像转换成 `star-enhanced map`。之后再在增强图上做峰值检测/DAOFIND/LoG。

这次实验用 `data_model` 中叠加图 mask 检测出的星点位置作为监督信号。训练时先把星点位置写成单像素 delta 图，再按 DeepSource 思路生成平滑 demand map。若要严格测试单像素 delta，可把训练参数改为 `--target-mode delta`。

本实验为独立实现，不复用 `00_unet_heatmap` 的 dataset、postprocess 或训练代码。FITS 读取、mask 读取、图像归一化、mask peak 提取和 target 生成都在 `code/deepsource_star/data.py` 中重新实现。

## 模型架构

照搬 DeepSource 小型全卷积 residual CNN：

```text
input 1ch image
Conv2d 5x5, 16, ReLU
Conv2d 5x5, 16, ReLU
Conv2d 5x5, 16, ReLU
residual add with first conv feature
BatchNorm
Conv2d 5x5, 16, ReLU
Dropout
Conv2d 5x5, 1, ReLU
output enhanced map
```

训练 loss 为 MSE，优化器为 RMSprop。默认参数：

- filters: `16`
- kernel size: `5`
- alpha: `0.75`
- background level: `0.05`
- target mode: `deepsource`

## 训练命令

默认数据划分与 01~10 的 `data_model/manifest.csv` 对齐：

- train: `split_reason=train`
- val: `split_reason=frame_holdout`
- test: `split_reason=coord_holdout`

默认每张图随机裁剪 `20` 个 `200x200` patch，训练、验证、测试三组都使用同一套裁剪逻辑。

训练时每个 epoch 都会显示 train、val、test 进度。验证阶段会把模型输出的增强图经过 `DAOFIND sigma=5.0` 检测，再和验证 target 峰值做 `4px` 半径匹配，输出 precision、recall 和 F1。checkpoint 仍按 `val_loss` 选择 best。

脚本默认会自动寻找数据目录：优先使用仓库根目录下与 `single_star_test` 平级的 `data_model`，不存在时回退到旧的 `single_star_test/data/data_model`。

为避免训练时反复读取 FITS、mask 和生成 target，推荐先预生成固定 crop 数据集：

```bash
python -u single_star_test/11_deepsource_star_enhancer/code/scripts/build_crop_dataset.py \
  --data-root data_model \
  --out-dir single_star_test/11_deepsource_star_enhancer/crop_data/deepsource_200x20_seed42 \
  --train-samples 0 \
  --val-samples 0 \
  --test-samples 0 \
  --crop-size 200 \
  --crops-per-image 20 \
  --target-mode deepsource \
  --seed 42
```

本地 smoke test：

```bash
python -u single_star_test/11_deepsource_star_enhancer/code/deepsource_star/train.py \
  --data-root data_model \
  --out-dir single_star_test/11_deepsource_star_enhancer/results/smoke_deepsource_seed42 \
  --train-samples 8 \
  --val-samples 2 \
  --test-samples 2 \
  --crop-size 200 \
  --crops-per-image 2 \
  --target-mode deepsource \
  --epochs 2 \
  --batch-size 2 \
  --device cuda \
  --seed 42
```

云端全量训练：

```bash
python -u single_star_test/11_deepsource_star_enhancer/code/deepsource_star/train.py \
  --data-root data_model \
  --crop-data-dir single_star_test/11_deepsource_star_enhancer/crop_data/deepsource_200x20_seed42 \
  --out-dir single_star_test/11_deepsource_star_enhancer/results/deepsource_aligned_seed42 \
  --train-samples 0 \
  --val-samples 0 \
  --test-samples 0 \
  --crop-size 200 \
  --crops-per-image 20 \
  --target-mode deepsource \
  --epochs 20 \
  --batch-size 128 \
  --num-workers 2 \
  --val-dao-method daofind:5.0 \
  --device cuda \
  --seed 42
```

## 当前要验证的问题

1. 这种小型增强网络能不能在 `data_model` 上稳定学到非零的星点增强图。
2. 增强图上跑峰值检测后，能否超过原图 DAOFIND sigma=4.0 的完整 holdout F1。
3. 如果 `deepsource` target 有效，再对比 `delta` target，验证 DeepSource 为什么不直接用单像素标签训练。
