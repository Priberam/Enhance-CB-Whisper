import torch

from transformers import ResNetConfig, ResNetModel
from typing import Optional


class Resnet(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: Optional[int] = None,
        version: str = "resnet-50",
    ):
        super().__init__()

        # save parameters
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.version = version

        # initializing a ResNet resnet-50 style configuration
        self.config = ResNetConfig()
        if self.version == "resnet-18":
            self.config.layer_type = "basic"
            self.config.hidden_sizes = [64, 128, 256, 512]
            self.config.depths = [2, 2, 2, 2]
        elif self.version == "resnet-34":
            self.config.layer_type = "basic"
            self.config.hidden_sizes = [64, 128, 256, 512]
            self.config.depths = [3, 4, 6, 3]

        # set number of input channels
        self.config.num_channels = self.num_channels
        if self.num_classes != None:
            # set number of classes
            self.config.num_labels = self.num_classes

        self.feature_extractor = ResNetModel(self.config)

        if self.num_classes != None:
            # a linear layer for kws classification
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1, end_dim=-1),
                torch.nn.Linear(
                    in_features=self.config.hidden_sizes[-1],
                    out_features=self.config.num_labels,
                    bias=True,
                ),
            )

    def forward(self, input_features: torch.Tensor):
        features = torch.flatten(
            self.feature_extractor(input_features).pooler_output,
            start_dim=1,
            end_dim=-1,
        )
        logits = self.classifier(features)
        return logits
