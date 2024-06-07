import torch


class ResNetDiscriminator(torch.nn.Module):
    def __init__(
        self, 
        in_features: int,
        num_labels: int,
        **kwargs
    ):
        super().__init__()

        self.in_features = in_features
        self.num_labels = num_labels

        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(in_features=self.in_features, out_features=self.num_labels, bias=True)
        )

    def forward(self, x):
        return self.layers(x)
    

class ResNetDiscriminatorLarge(torch.nn.Module):
    def __init__(
        self, 
        in_features: int,
        num_labels: int,
        **kwargs
    ):
        super().__init__()

        self.in_features = in_features
        self.num_labels = num_labels
        self.hidden_dim = in_features // 2

        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(in_features=self.in_features, out_features=self.hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True),
            torch.nn.ReLU(), 
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=self.hidden_dim, out_features=self.num_labels, bias=True)
        )

    def forward(self, x):
        return self.layers(x)