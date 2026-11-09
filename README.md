# RRSI

**English** | [中文](README.zh-CN.md)

## Radiation, rotation and scale invariant feature descriptor for multimodal image matching

Inference code for RRSI, published in **Information Fusion**, Volume 139 (2027), Article 104903. [Paper](https://doi.org/10.1016/j.inffus.2026.104903)

## Overview

RRSI is a feature descriptor for multimodal image matching under radiometric, rotation, and scale variations. This package provides an optical–SAR inference interface that takes two images and saves a matching visualization.

- **Image A** — First input modality.
- **Image B** — Second input modality.
- **Output** — Side-by-side images with yellow lines connecting geometrically filtered matches.

## Qualitative Results

<!-- Matching visualization -->

## Model Zoo

Download the optical–SAR checkpoint and place it at `weights/rrsi.pt`.

| Modality | Filename | Google Drive | Baidu Netdisk |
| --- | --- | --- | --- |
| Optical–SAR | `rrsi.pt` |[[Google Drive Link]](https://drive.google.com/file/d/1rtvVAcAjwWbNmRaiVQS5VD6OK42m_fLS/view?usp=sharing)|[[Baidu Netdisk Link]](https://pan.baidu.com/s/1-RJtKPIxitpkqpoK9LX8xA?pwd=if44)|

## Installation

Use Python 3.8 or later. Install a PyTorch build suitable for your CPU or CUDA device, then install the dependencies from the repository directory.

```bash
pip install -r requirements.txt
```

## Demo

### Step 1 — Prepare the checkpoint

Place the checkpoint from the Model Zoo in the following location.

```text
weights/
└── rrsi.pt
```

### Step 2 — Prepare your images

Pass your two modality images to `--image-a` and `--image-b`. The input order can be exchanged. Grayscale and color images are supported, and the two images may have different dimensions. Paths may contain spaces or Unicode characters.

### Step 3 — Run inference

```bash
python infer.py --image-a image_a.png --image-b image_b.png --output matches.png --show
```

Replace `image_a.png` and `image_b.png` with your image paths. The matching visualization is saved to `matches.png`. The `--show` option displays the saved image, and pressing a key in the image window closes it.

The console prints the candidate match count, retained match count, and saved file path. A failed match prints an explanatory message and returns exit code 1.

<details>
<summary>More options</summary>

| Option | Default | Purpose |
| --- | --- | --- |
| `--image-a` | Required | First input image path |
| `--image-b` | Required | Second input image path |
| `--output` | `matches.png` | Output image path |
| `--checkpoint` | Package `weights/rrsi.pt` | Checkpoint path |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `cuda:N` |
| `--max-keypoints` | `7500` | Keypoint budget per image |
| `--batch-size` | `128` | Patch sampling batch size |
| `--show` | Off | Display the saved image |

`auto` selects CUDA when available and CPU otherwise. To select CPU explicitly, run the following command.

```bash
python infer.py --image-a "path/to/image_a.png" --image-b "path/to/image_b.png" --output matches.png --device cpu
```

</details>

## Citation

If this work is useful for your research, please cite the paper.

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

## Contact

- Tengfeng Tang (Corresponding Author) — `ttf@my.swjtu.edu.cn`
- Yuanxin Ye (First Author) — `yeyuanxin@home.swjtu.edu.cn`
