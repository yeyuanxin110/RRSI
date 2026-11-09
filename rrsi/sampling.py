"""Log-Polar head sampled from Cartesian patches."""

import torch
import torch.nn.functional as F


def cartesian_to_logpolar(patches: torch.Tensor) -> torch.Tensor:
    count, _, height, width = patches.shape
    device = patches.device
    dtype = patches.dtype
    radial_axis = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    angle_axis = torch.linspace(0.0, 2.0 * torch.pi, height + 1, device=device, dtype=dtype)[:-1]
    theta, radial = torch.meshgrid(angle_axis, radial_axis, indexing="ij")
    max_radius = torch.tensor(min(height, width) / 2.0, device=device, dtype=dtype)
    radius = torch.expm1(radial * torch.log(max_radius)) / (max_radius - 1.0)
    grid_x = radius * torch.cos(theta)
    grid_y = radius * torch.sin(theta)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(count, -1, -1, -1)
    return F.grid_sample(patches, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
