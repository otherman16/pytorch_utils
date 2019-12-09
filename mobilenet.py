import torch.nn as nn


class Conv2dBNReLU6(nn.Sequential):
    '''Convolution2d + BatchNormalization2d + ReLU6 Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(Conv2dBNReLU6, self).__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class Conv2dBN(nn.Sequential):
    '''Convolution2d + BatchNormalization2d'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(Conv2dBN, self).__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )


class DWSConv(nn.Sequential):
    '''Depthwise Separable Convolution'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DWSConv, self).__init__(
            Conv2dBNReLU6(in_channels, in_channels,
                          kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels),
            Conv2dBNReLU6(in_channels, out_channels,
                          kernel_size=1)
        )


class DOLinear(nn.Sequential):
    '''Droupout + Linear'''
    def __init__(self, in_features, out_features, bias=True):
        super(DOLinear, self).__init__(
            nn.Dropout(0.2),
            nn.Linear(in_features, out_features, bias=bias)
        )


class MobileNet(nn.Module):

    CFG = [
        # c, s
        [64, 1],
        [128, 2],
        [128, 1],
        [256, 2],
        [256, 1],
        [512, 2],
        [512, 1],
        [512, 1],
        [512, 1],
        [512, 1],
        [512, 1],
        [1024, 2],
        [1024, 1]
    ]

    def __init__(self, width_multiplier=1.0, in_channels=3, classes=1000):
        super(MobileNet, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.cfg = [[int(c*width_multiplier), s] for c, s in MobileNet.CFG]

        self.feature_extractor = self.make_feature_extractor()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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
        x = x.view(-1, 1 * 1 * self.cfg[-1][0])
        x = self.classifier(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x

    def make_feature_extractor(self):
        layers = [Conv2dBNReLU6(self.in_channels, 32, kernel_size=3, stride=2, padding=1)]

        in_channels = 32
        for out_channels, stride in self.cfg:
            layers.append(DWSConv(in_channels, out_channels, kernel_size=3, stride=stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def make_classifier(self):
        return DOLinear(1 * 1 * self.cfg[-1][0], self.classes)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expand_ratio=6.0):
        super(InvertedResidual, self).__init__()

        self.res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = [
            Conv2dBNReLU6(in_channels, hidden_dim,
                          kernel_size=1),
            Conv2dBNReLU6(hidden_dim, hidden_dim,
                          kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim),
            Conv2dBN(hidden_dim, out_channels, kernel_size=1, bias=False)
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.res_connect:
            return self.conv(x) + x
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    CFG = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]

    def __init__(self, width_multiplier=1.0, in_channels=3, classes=1000):
        super(MobileNetV2, self).__init__()
        self.in_channels = in_channels
        self.classes = classes

        self.cfg = [[t, int(round(c*width_multiplier)), n, s] for t, c, n, s in MobileNetV2.CFG]

        self.feature_extractor = self.make_feature_extractor()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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
        x = x.view(-1, 1 * 1 * 1280)
        x = self.classifier(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x

    def make_feature_extractor(self):
        layers = [Conv2dBNReLU6(self.in_channels, 32, kernel_size=3, stride=2, padding=1)]

        in_channels = 32
        for t, c, n, s in self.cfg:
            for i in range(n):
                s = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, c, kernel_size=3, stride=s, expand_ratio=t))
                in_channels = c

        layers.append(Conv2dBNReLU6(in_channels, 1280, kernel_size=1))

        return nn.Sequential(*layers)

    def make_classifier(self):
        return DOLinear(1 * 1 * 1280, self.classes)
