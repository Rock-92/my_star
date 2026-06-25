# 00 U-Net Heatmap

## 目的

直接从单帧图像预测星点 heatmap，再提取峰值坐标。

## 架构

U-Net 风格 heatmap 分割网络，代码在 `code/star_unet/`。

## 结果

低阈值时假阳性爆炸，高阈值时漏检严重，整体不如后来的“低阈值候选 + CNN scorer”稳定，因此没有作为最终主路线继续推进。

## 结果文件

`results/star_unet/` 保存历史 checkpoint、训练记录和 DAOFIND 对比图。
