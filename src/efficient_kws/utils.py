from dataclasses import dataclass
import torch


@dataclass
class KWSOutput:
    """Class for storing the output of the KWS model."""

    logits: torch.Tensor
    features: torch.Tensor
    loss: float = None
    logits_alt: torch.Tensor = None
    loss_alt: float = None
