import torch
from .utils import DiscOutput
"""
GRL adpated from Matsuura et al. 2020
(Code) https://github.com/mil-tokyo/dg_mmld/
(Paper) https://arxiv.org/pdf/1911.07661.pdf 
"""


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, reverse=True):
        ctx.beta = beta
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.beta), None, None
        else:
            return (grad_output * ctx.beta), None, None


def grad_reverse(x, beta=1.0, reverse=True):
    return GradReverse.apply(x, beta, reverse)


class Discriminator(torch.nn.Module):
    def __init__(
        self, 
        head: torch.nn.Module, 
        reverse: bool = True,
        **kwargs
    ):
        super().__init__()
        self.head = head
        self.beta = 0.0
        self.reverse = reverse

    def set_beta(
        self, 
        beta: float
    ):
        self.beta = beta

    def forward(
        self, 
        input_features: torch.Tensor, 
        labels: torch.Tensor = None,
        use_grad_reverse: bool = True
    ):
        if use_grad_reverse:
            input_features_ = grad_reverse(input_features, self.beta, reverse=self.reverse)
        else:
            input_features_ = input_features
        logits = self.head(input_features_)
        if labels != None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.head.num_labels), labels.view(-1))
        else:
            loss = None
        return DiscOutput(
            logits = logits,
            loss = loss
        )