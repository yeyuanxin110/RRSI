# RRSI

[English](README.md) | **中文**

## Radiation, rotation and scale invariant feature descriptor for multimodal image matching

本仓库提供 RRSI 的推理代码。论文发表于 **Information Fusion**，第139卷（2027年），文章编号104903。[论文链接](https://doi.org/10.1016/j.inffus.2026.104903)

## 简介

RRSI 是面向辐射差异、旋转与尺度变化的多模态图像匹配特征描述子。本推理包提供光学–SAR匹配接口，输入两张图像，保存匹配连线可视化结果。

- **图像 A** — 第一种模态图像。
- **图像 B** — 第二种模态图像。
- **输出** — 两张图像左右并排显示，用黄色线段连接经过几何筛选的匹配点。

## 匹配示例

<!-- 匹配可视化图片 -->

## 模型权重

下载光学–SAR模型权重，并放在 `weights/rrsi.pt`。

| 模态 | 文件名 | 谷歌云盘 | 百度网盘 |
| --- | --- | --- | --- |
| 光学–SAR | `rrsi.pt` | | |

## 环境安装

使用 Python 3.8 或更高版本。安装适合本机 CPU 或 CUDA 环境的 PyTorch，然后在仓库目录中安装依赖。

```bash
pip install -r requirements.txt
```

## Demo使用

### 第一步 — 准备模型权重

将上方模型权重放在以下位置。

```text
weights/
└── rrsi.pt
```

### 第二步 — 准备图像

将两种模态的图像路径分别传给 `--image-a` 和 `--image-b`，二者可以交换顺序。支持灰度图和彩色图，两张图像的尺寸可以不同。图像路径支持空格及中文等 Unicode 字符。

### 第三步 — 运行推理

```bash
python infer.py --image-a image_a.png --image-b image_b.png --output matches.png --show
```

将 `image_a.png` 和 `image_b.png` 替换为自己的图像路径。匹配可视化结果保存至 `matches.png`。`--show` 用于显示保存后的匹配图，在图像窗口中按任意键即可关闭窗口。

控制台输出候选匹配数量、保留的匹配数量和保存路径。匹配失败时会输出原因，并返回退出码 1。

<details>
<summary>更多参数</summary>

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--image-a` | 必填 | 第一张输入图像的路径 |
| `--image-b` | 必填 | 第二张输入图像的路径 |
| `--output` | `matches.png` | 输出图像路径 |
| `--checkpoint` | 推理包内的 `weights/rrsi.pt` | 模型权重路径 |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或 `cuda:N` |
| `--max-keypoints` | `7500` | 每张图像的关键点数量上限 |
| `--batch-size` | `128` | 每次采样的图像块数量 |
| `--show` | 关闭 | 显示保存后的匹配图 |

`auto` 在 CUDA 可用时使用 CUDA，否则使用 CPU。如需指定使用 CPU，可运行以下命令。

```bash
python infer.py --image-a "path/to/image_a.png" --image-b "path/to/image_b.png" --output matches.png --device cpu
```

</details>

## 引用

如果本项目对您的研究有所帮助，请引用以下论文。

```bibtex
@article{ye2027rrsi,
  title   = {Radiation, rotation and scale invariant feature descriptor for multimodal image matching},
  author  = {Ye, Yuanxin and Tang, Tengfeng and Peng, Tao and Han, Zhiqiang and Li, Jiayuan and Wang, Mi},
  journal = {Information Fusion},
  volume  = {139},
  pages   = {104903},
  year    = {2027},
  doi     = {10.1016/j.inffus.2026.104903}
}
```

## 联系方式

- 唐腾峰（通讯作者）— `ttf@my.swjtu.edu.cn`
- 叶沅鑫 教授（第一作者）— `yeyuanxin@home.swjtu.edu.cn`
