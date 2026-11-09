"""STN image and keypoint preprocessing."""

import torch


def preprocess_Transformer_layer(im, kps, image_name, device):
    img = torch.as_tensor(im).to(device).unsqueeze(0)
    height, width = im.shape
    # Normalize keypoint coordinates after square padding.
    kps = kps.copy()
    kps[:, 0] = (kps[:, 0] - height / 2) / (height / 2)
    kps[:, 1] = (kps[:, 1] - width / 2) / (width / 2)
    theta = [
        torch.as_tensor(kps).to(device),
        torch.as_tensor([1]).to(device),
        torch.as_tensor([0]).to(device),
    ]
    return [{image_name: img}, theta, [image_name] * len(kps)]
