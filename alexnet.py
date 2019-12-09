import torch.nn as nn


class Conv2dBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DOLinearReLU(nn.Sequential):

    def __init__(self, in_features, out_features):
        super(DOLinearReLU, self).__init__(
            nn.Dropout(),
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )


class DOLinear(nn.Sequential):

    def __init__(self, in_features, out_features):
        super(DOLinear, self).__init__(
            nn.Dropout(),
            nn.Linear(in_features, out_features)
        )


class AlexNet(nn.Module):

    def __init__(self, in_channels=3, classes=1000):
        super(AlexNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            Conv2dBNReLU(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            Conv2dBNReLU(96, 256, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            Conv2dBNReLU(256, 384, kernel_size=3, stride=1, padding=1),
            Conv2dBNReLU(384, 384, kernel_size=3, stride=1, padding=1),
            Conv2dBNReLU(384, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            DOLinearReLU(6 * 6 * 256, 4096),
            DOLinearReLU(4096, 4096),
            DOLinear(4096, classes)
        )
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
        x = x.view(-1, 6 * 6 * 256)
        x = self.classifier(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x
