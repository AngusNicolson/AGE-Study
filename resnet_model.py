
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet, _resnet
from torch import Tensor


class BasicBlockDropout(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.03
    ) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.dropout_layer(out)

        return out


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlockDropout, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet18_wrapper(pretrained: bool = False, progress: bool = True, dropout_2d=False, **kwargs):
    if dropout_2d:
        return resnet18(pretrained, progress, **kwargs)
    else:
        return models.resnet18(pretrained, progress, **kwargs)


if __name__ == "__main__":
    model = resnet18(pretrained=True)
    x = torch.empty((1, 3, 224, 224))
    out = model(x)
    print(out.shape)

