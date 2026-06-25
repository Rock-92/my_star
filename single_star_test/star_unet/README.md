# U-Net 单帧 FITS 星点增强

这个模块用于训练一个单输出 U-Net，让模型从单帧 FITS 星图中预测星点高斯热图。当前目标不是直接保证达到 20 帧叠加图的完整效果，而是先建立一个稳定 baseline，证明模型能比传统单帧提取方法看到更多弱星点。

当前第一版目标：

```text
传统 DAOFind-like 单帧基线：约 G=12.0
第一目标：U-Net 单帧可靠极限提升到 G=13.0-13.5
挑战目标：逼近叠加图的 G=14.0
```

## 当前方案

数据生成使用 `preprocessing/model_data_processing.py`：

- 输入：`data/data_S30Pro/*_sub/*.fit`
- 标签：对应叠加图用 `tetra3_like` 提取星点
- 标签形式：高斯热图，不是二值 mask
- 高斯半径：tetra3 圆盘估计半径扩大 `2.0` 倍
- 对齐方式：只用 FITS 头里的 RA/DEC 和叠加图 WCS 做整体平移，不依赖 Gaia
- 输出目录：`data/data_model/train` 和 `data/data_model/val`

验证集划分：

```text
floor(坐标数 * 10%) 个坐标整组进入 val
剩余坐标中，每个 *_sub 再抽 floor(单帧数 * 10%) 进入 val
```

`manifest.csv` 会记录每个样本的 `split_reason`：

```text
train
frame_holdout
coord_holdout
```

## 数据格式

期望的数据结构：

```text
data/data_model/
  train/
    images/sample_000001.fit
    masks/sample_000001.png
  val/
    images/sample_000010.fit
    masks/sample_000010.png
  manifest.csv
  summary.json
```

输入图像可以是 FITS 或普通位图。当前默认按单通道读取 FITS：

```json
"fit_channel_mode": "mean"
```

FITS 会按每张图做 percentile 归一化：

```json
"image_normalization": {
  "mode": "percentile",
  "lower_percentile": 0.5,
  "upper_percentile": 99.8
}
```

mask 是 0-1 连续热图标签，训练时不会再二值化。

## 训练

采集完数据后，先生成训练数据：

```powershell
python preprocessing/model_data_processing.py --overwrite
```

然后训练第一版模型：

```powershell
python -m star_unet.train --config star_unet/config.json
```

当前默认训练配置：

```text
模型：单通道 U-Net
features：[32, 64, 128]，总下采样 4 倍
输入：单帧 FITS -> mean 灰度 -> 0-1 归一化
输出：1 通道 heatmap logits
loss：BCEWithLogits + DiceLoss
positive_weight：30
batch_size：1
训练 crop：1024x1024 random crop
验证/推理：整图 tiled prediction
```

2160x3840 整图直接训练显存压力较大，所以训练时默认随机裁剪 1024x1024 patch。验证和推理时会用 tiled inference 拼回整张图。

## 预测和评估

对 FITS 文件或目录推理：

```powershell
python -m star_unet.predict data/data_model/val/images --checkpoint runs/star_unet/best.pt --output runs/star_unet/predictions
```

在验证集上评估 U-Net 和传统方法：

```powershell
python -m star_unet.evaluate --checkpoint runs/star_unet/best.pt --out-dir runs/star_unet/eval
```

如果还没有训练 checkpoint，可以只评估传统基线：

```powershell
python -m star_unet.evaluate --skip-model --out-dir runs/star_unet/eval_baselines
```

评估会比较：

```text
tetra3_like
sextractor_like
daofind_like
U-Net threshold = 0.2, 0.3, 0.4, 0.5
```

默认没有 Gaia 时，评估以叠加图生成的高斯热图标签作为伪真值；如果某些视场有 Gaia，可以后续再做更严格的星等分箱评估。

## 后续实验路线

第一阶段先跑 baseline：

```text
单通道 mean FITS
高斯半径倍率 2.0
positive_weight = 30
归一化 = 0.5-99.8 percentile
训练 100 epoch
```

如果 baseline 跑通，再做参数扫描：

```powershell
python -m star_unet.train --positive-weight 10 --out-dir runs/star_unet/pos10
python -m star_unet.train --positive-weight 50 --out-dir runs/star_unet/pos50
python -m star_unet.train --positive-weight 100 --out-dir runs/star_unet/pos100
```

归一化对照实验：

```powershell
python -m star_unet.train --norm-lower-percentile 0.1 --norm-upper-percentile 99.9 --out-dir runs/star_unet/norm_0p1_99p9
python -m star_unet.train --norm-lower-percentile 1.0 --norm-upper-percentile 99.5 --out-dir runs/star_unet/norm_1p0_99p5
```

如果单通道 baseline 达到瓶颈，再考虑 3 通道 FITS 输入。第一版暂时不改 3 通道，避免一次引入太多变量。
