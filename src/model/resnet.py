import torch
from transformers import ResNetConfig, ResNetModel


class Resnet(torch.nn.Module):
    def __init__(
        self, 
        num_channels: int,
        num_classes: int
    ):
        super().__init__()

        # initializing a ResNet resnet-50 style configuration
        self.config = ResNetConfig()
        # set number of input channels to 12
        self.config.num_channels = num_channels
        # set number of classes to 2
        self.config.num_labels = num_classes

        # initializing a model (with random weights) from the resnet-50 style configuration for feature extraction
        self.feature_extractor = ResNetModel(self.config)
        # a linear layer for kws classification
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(in_features=self.config.hidden_sizes[-1], out_features=self.config.num_labels, bias=True)
        )

    def forward(
        self, 
        input_features: torch.Tensor
    ):
        features = self.feature_extractor(pixel_values = input_features).pooler_output
        logits = self.classifier(features)
        return logits