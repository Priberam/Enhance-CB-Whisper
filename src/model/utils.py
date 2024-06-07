from dataclasses import dataclass
import torch

@dataclass
class KWSOutput:
    """Class for storing the output of the KWS model."""
    logits: torch.Tensor
    features: torch.Tensor
    loss: float = None

@dataclass
class DiscOutput:
    """Class for storing the output of the Discriminator model."""
    logits: torch.Tensor
    loss: float = None