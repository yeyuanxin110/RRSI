"""Load a descriptor checkpoint for paired-image inference."""

from pathlib import Path

import torch

from rrsi.model import RRSIDescriptor


def load_descriptor(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_config" not in payload or "descriptor_state" not in payload:
        raise ValueError("Expected a descriptor checkpoint with model_config and descriptor_state.")
    model = RRSIDescriptor(payload["model_config"])
    model.load_state_dict(payload["descriptor_state"], strict=True)
    return model.to(device).eval()
