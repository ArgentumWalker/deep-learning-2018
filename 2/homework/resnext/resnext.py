import torch.nn as nn
from torch import load, save
from torchvision.models.resnet import ResNet

__all__ = ['resnext50', 'resnext101', 'resnext152', 'save_resnext']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=32):
        super(Bottleneck, self).__init__()
        sub_planes = planes * 2
        self.conv1 = nn.Conv2d(inplanes, sub_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(sub_planes)
        self.conv2 = nn.Conv2d(sub_planes, sub_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(sub_planes)
        self.conv3 = nn.Conv2d(sub_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnext50(pretrained=None, **kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained is not None:
        model.load_state_dict(load(pretrained))
    return model


def resnext101(pretrained=None, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained is not None:
        model.load_state_dict(load(pretrained))
    return model


def resnext152(pretrained=None, **kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained is not None:
        model.load_state_dict(load(pretrained))
        model.eval()
    return model


def save_resnext(path, model):
    save(model.state_dict(), path)