"""256-dimensional patch descriptor."""

import torch
import torch.nn as nn


eps_l2_norm = 1e-10


def desc_l2norm(descriptors):
    return descriptors / descriptors.pow(2).sum(dim=1, keepdim=True).add(
        eps_l2_norm
    ).pow(0.5)


class FRN(nn.Module):
    def __init__(
        self, num_features, eps=1e-6, is_bias=True, is_scale=True,
        is_eps_leanable=False,
    ):
        super(FRN, self).__init__()
        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale
        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        if is_eps_leanable:
            self.eps = nn.Parameter(torch.Tensor(1))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return "num_features={num_features}, eps={init_eps}".format(**self.__dict__)

    def forward(self, values):
        mean_square = values.pow(2).mean(dim=[2, 3], keepdim=True)
        values = values * torch.rsqrt(mean_square + self.eps.abs())
        if self.is_scale:
            values = self.weight * values
        if self.is_bias:
            values = values + self.bias
        return values


class TLU(nn.Module):
    def __init__(self, num_features):
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return "num_features={num_features}".format(**self.__dict__)

    def forward(self, values):
        return torch.max(values, self.tau)


class CNN_model1(nn.Module):
    def __init__(
        self, is_bias=True, is_bias_FRN=True, input_size=128,
        dim_desc=256, drop_rate=0.2,
    ):
        super(CNN_model1, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(input_size),
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer10 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(256, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False),
        )

    def forward(self, patches):
        for layer in (
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer5,
            self.layer6, self.layer7, self.layer8, self.layer9,
        ):
            patches = layer(patches)
        raw_descriptors = self.layer10(patches).flatten(1)
        descriptors = desc_l2norm(raw_descriptors)
        return descriptors
