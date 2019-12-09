import torch.nn as nn
import torch.nn.functional as F


class Conv2dBNReLU(nn.Sequential):
    '''Convolution2d + BatchNormalization2d + ReLU Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Conv2dBN(nn.Sequential):
    '''Convolution2d + BatchNormalization2d'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(Conv2dBN, self).__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )


class DOLinearBNReLU(nn.Sequential):
    '''Droupout + Linear + BatchNormalization1d + ReLU Activation'''
    def __init__(self, in_features, out_features, bias=True):
        super(DOLinearBNReLU, self).__init__(
            nn.Dropout(0.2),
            nn.Linear(in_features, out_features, bias=bias),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )


class DOLinear(nn.Sequential):
    '''Droupout + Linear'''
    def __init__(self, in_features, out_features, bias=True):
        super(DOLinear, self).__init__(
            nn.Dropout(0.2),
            nn.Linear(in_features, out_features, bias=bias)
        )


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()

        self.need_downsample = in_channels != out_channels

        if self.need_downsample:
            self.downsample = Conv2dBN(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        self.conv = nn.Sequential(
            Conv2dBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            Conv2dBN(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        if self.need_downsample:
            return F.relu(self.conv(x) + self.downsample(x))
        else:
            return F.relu(self.conv(x) + x)


class ResidualBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels):
        super(ResidualBottleneckBlock, self).__init__()
        self.res_connection = in_channels == out_channels

        self.conv = nn.Sequential(
            Conv2dBNReLU(in_channels, out_channels, kernel_size=1, bias=False),
            Conv2dBNReLU(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Conv2dBN(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)
        )

    def forward(self, x):
        if self.res_connection:
            return F.relu(self.conv(x) + x)
        else:
            return F.relu(self.conv(x))


class ResNet(nn.Module):

    CFGS = {
        '18': [2, 2, 2, 2],
        '34': [3, 4, 6, 3],
        '50': [3, 4, 6, 3],
        '101': [3, 4, 23, 3],
        '152': [3, 8, 36, 3],
    }

    def __init__(self, in_channels=3, classes=1000, cfg='18'):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.cfg = ResNet.CFGS[cfg]
        self.block = ResidualBlock if cfg in ('18', '34') else ResidualBottleneckBlock

        self.feature_extractor = self.make_feature_extractor()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self.make_classifier()

        self.sm = nn.Softmax

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(-1, 1 * 1 * 512 * self.block.expansion)
        x = self.classifier(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x

    def make_feature_extractor(self):
        layers = [
            Conv2dBNReLU(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        in_channels = 64
        out_channels = 64
        for n in self.cfg:
            for i in range(n):
                stride = 2 if i == 0 and in_channels != 64 else 1
                layers.append(self.block(in_channels*self.block.expansion, out_channels, stride))
                in_channels = out_channels
            out_channels *= 2
        return nn.Sequential(*layers)

    def make_classifier(self):
        return DOLinear(1 * 1 * 512 * self.block.expansion, self.classes)
