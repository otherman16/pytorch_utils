import torch.nn as nn


class Conv2dBNReLU(nn.Sequential):
    '''Convolution2d + BatchNormalization2d + ReLU Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DOLinearBNReLU(nn.Sequential):
    '''Droupout + Linear + BatchNormalization1d + ReLU Activation'''
    def __init__(self, in_features, out_features, bias=True):
        super(DOLinearBNReLU, self).__init__(
            nn.Dropout(),
            nn.Linear(in_features, out_features, bias=bias),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )


class DOLinear(nn.Sequential):
    '''Droupout + Linear'''
    def __init__(self, in_features, out_features, bias=True):
        super(DOLinear, self).__init__(
            nn.Dropout(),
            nn.Linear(in_features, out_features, bias=bias)
        )


class VGG(nn.Module):

    CFGS = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, in_channels=3, classes=1000, cfg='A'):
        super(VGG, self).__init__()

        self.in_channels = in_channels
        self.classes = classes
        self.cfg = VGG.CFGS[cfg]

        self.feature_extractor = self.make_feature_extractor()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self.make_classifier()

        self.sm = nn.Softmax()

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
        x = x.view(-1, 7 * 7 * self.cfg[-2])
        x = self.classifier(x)

        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x

    def make_feature_extractor(self):
        layers = []

        in_channels = self.in_channels
        for out_channels in self.cfg:
            if out_channels == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(Conv2dBNReLU(in_channels, out_channels, kernel_size=3, padding=1))
                in_channels = out_channels

        return nn.Sequential(*layers)

    def make_classifier(self):
        return nn.Sequential(
            DOLinearBNReLU(7 * 7 * self.cfg[-2], 4096),
            DOLinearBNReLU(4096, 4096),
            DOLinear(4096, self.classes),
        )
