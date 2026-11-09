"""Cartesian STN patch sampling for inference."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self, transform="STN", coords="linear", resolution=32,
        SIFTscale=12.0, onGPU=True, squareSize=1500.0,
    ):
        super(Transformer, self).__init__()
        if transform != "STN" or coords != "linear":
            raise ValueError("This inference package uses Cartesian STN sampling.")
        self.resolution = resolution
        self.SIFTscale = SIFTscale
        self.rescaleSTNtoSIFT = 1 / squareSize

    def getGridSize(self, image, batchSize=1):
        shape = list(image.shape)
        shape[-2:] = [self.resolution, self.resolution]
        shape[0] = batchSize
        return torch.Size(shape)

    def stn(self, image, theta, image_ids=None):
        locations, scales, rotations = theta
        scales = self.rescaleSTNtoSIFT * self.SIFTscale * scales
        batch_size = locations.shape[0]
        device = locations.device

        # Construct affine grids in CPU float32.
        affine = torch.empty(batch_size, 2, 3, dtype=torch.float32)
        affine[:, 0, 0] = torch.cos(rotations) * scales
        affine[:, 0, 1] = -scales * torch.sin(rotations)
        affine[:, 0, 2] = locations[:, 0]
        affine[:, 1, 0] = scales * torch.sin(rotations)
        affine[:, 1, 1] = torch.cos(rotations) * scales
        affine[:, 1, 2] = locations[:, 1]

        if isinstance(image, dict):
            patches = torch.empty(
                batch_size, 1, self.resolution, self.resolution, device=device,
            )
            for key, values in image.items():
                values = values.unsqueeze(0).to(device)
                indices = [
                    index for index, image_id in enumerate(image_ids)
                    if image_id == key
                ]
                if not indices:
                    continue
                root = np.sqrt(len(indices))
                if root % 1 == 0:
                    grid = F.affine_grid(
                        affine[indices], self.getGridSize(values, len(indices)),
                        align_corners=False,
                    )
                    side = int(self.resolution * root)
                    grid = grid.view(1, side, side, 2)
                    sampled = F.grid_sample(
                        values.float(), grid.to(device).float(), align_corners=False,
                    )
                    patches[indices] = sampled.view(
                        len(indices), 1, self.resolution, self.resolution,
                    )
                else:
                    for index in indices:
                        grid = F.affine_grid(
                            affine[index].unsqueeze(0), self.getGridSize(values),
                            align_corners=False,
                        )
                        patches[index, 0] = F.grid_sample(
                            values.float(), grid.to(device).float(),
                            align_corners=False,
                        )
            return patches

        grid = F.affine_grid(
            affine, self.getGridSize(image, batch_size), align_corners=False,
        )
        return F.grid_sample(
            image.to(device).float(), grid.to(device).float(), align_corners=False,
        )

    def forward(self, inputs):
        image, theta = inputs[0], inputs[1]
        image_ids = None if len(inputs) < 3 else inputs[2]
        if theta[0].shape[0] == 0:
            return torch.empty(
                0, 1, self.resolution, self.resolution, device=theta[0].device,
            )
        return self.stn(image, theta, image_ids) / 255.0
