import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls


__all__ = ['FPNResNet', 'fpn_resnet18', 'fpn_resnet34', 'fpn_resnet50', 'fpn_resnet101',
           'fpn_resnet152', 'fpn_resnext50_32x4d', 'fpn_resnext101_32x8d',
           'fpn_wide_resnet50_2', 'fpn_wide_resnet101_2']


class FPNResNet(nn.Module):

    def __init__(self, resnet, block, classes):
        super(FPNResNet, self).__init__()

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.conv2 = resnet.layer1
        self.conv3 = resnet.layer2
        self.conv4 = resnet.layer3
        self.conv5 = resnet.layer4

        self.lateral2 = nn.Sequential(
            nn.Conv2d(64*block.expansion, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        self.lateral3 = nn.Sequential(
            nn.Conv2d(128*block.expansion, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        self.lateral4 = nn.Sequential(
            nn.Conv2d(256*block.expansion, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        self.lateral5 = nn.Sequential(
            nn.Conv2d(512*block.expansion, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )

        self.dealiasing2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.dealiasing3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.dealiasing4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(56 * 56 * 256, classes)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        m5 = self.lateral5(c5)
        m4 = self.lateral4(c4) + F.interpolate(m5, size=(c4.shape[2], c4.shape[3]), mode='nearest')
        m3 = self.lateral3(c3) + F.interpolate(m4, size=(c3.shape[2], c3.shape[3]), mode='nearest')
        m2 = self.lateral2(c2) + F.interpolate(m3, size=(c2.shape[2], c2.shape[3]), mode='nearest')

        # p5 = m5
        # p4 = self.dealiasing4(m4)
        # p3 = self.dealiasing3(m3)
        p2 = self.dealiasing2(m2)

        out = p2.view(-1, 56 * 56 * 256)
        out = self.classifier(out)

        return out


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model


def _fpn_resnet(arch, block, layers, classes, pretrained, progress, **kwargs):
    model = _resnet(arch, block, layers, pretrained, progress, **kwargs)
    return FPNResNet(model, block, classes)


def fpn_resnet18(classes=1000, pretrained=False, progress=True, **kwargs):
    return _fpn_resnet('resnet18', BasicBlock, [2, 2, 2, 2], classes, pretrained, progress, **kwargs)


def fpn_resnet34(classes=1000, pretrained=False, progress=True, **kwargs):
    return _fpn_resnet('resnet34', BasicBlock, [3, 4, 6, 3], classes, pretrained, progress, **kwargs)


def fpn_resnet50(classes=1000, pretrained=False, progress=True, **kwargs):
    return _fpn_resnet('resnet50', Bottleneck, [3, 4, 6, 3], classes, pretrained, progress, **kwargs)


def fpn_resnet101(classes=1000, pretrained=False, progress=True, **kwargs):
    return _fpn_resnet('resnet101', Bottleneck, [3, 4, 23, 3], classes, pretrained, progress, **kwargs)


def fpn_resnet152(classes=1000, pretrained=False, progress=True, **kwargs):
    return _fpn_resnet('resnet152', Bottleneck, [3, 8, 36, 3], classes, pretrained, progress, **kwargs)


def fpn_resnext50_32x4d(classes=1000, pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _fpn_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], classes, pretrained, progress, **kwargs)


def fpn_resnext101_32x8d(classes=1000, pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _fpn_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], classes, pretrained, progress, **kwargs)


def fpn_wide_resnet50_2(classes=1000, pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _fpn_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], classes, pretrained, progress, **kwargs)


def fpn_wide_resnet101_2(classes=1000, pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _fpn_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], classes, pretrained, progress, **kwargs)
