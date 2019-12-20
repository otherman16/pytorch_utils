import torch
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


class InceptionBlockV1(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlockV1, self).__init__()
        self.branch1 = Conv2dBNReLU(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            Conv2dBNReLU(in_channels, ch3x3red, kernel_size=1),
            Conv2dBNReLU(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            Conv2dBNReLU(in_channels, ch5x5red, kernel_size=1),
            Conv2dBNReLU(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            Conv2dBNReLU(in_channels, pool_proj, kernel_size=1)
        )


class InceptionBlockV2(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlockV2, self).__init__()
        self.branch1 = Conv2dBNReLU(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            Conv2dBNReLU(in_channels, ch3x3red, kernel_size=1),
            Conv2dBNReLU(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            Conv2dBNReLU(in_channels, ch5x5red, kernel_size=1),
            Conv2dBNReLU(ch5x5red, ch5x5, kernel_size=3, padding=1),
            Conv2dBNReLU(ch5x5, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            Conv2dBNReLU(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class InceptionBlockV3(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlockV3, self).__init__()
        self.branch1 = Conv2dBNReLU(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            Conv2dBNReLU(in_channels, ch3x3red, kernel_size=1),
            Conv2dBNReLU(ch3x3red, ch3x3, kernel_size=(1, 3), padding=1),
            Conv2dBNReLU(ch3x3, ch3x3, kernel_size=(3, 1), padding=1)
        )

        self.branch3 = nn.Sequential(
            Conv2dBNReLU(in_channels, ch5x5red, kernel_size=1),
            Conv2dBNReLU(ch5x5red, ch5x5, kernel_size=(1, 3), padding=1),
            Conv2dBNReLU(ch5x5, ch5x5, kernel_size=(3, 1), padding=1),
            Conv2dBNReLU(ch5x5, ch5x5, kernel_size=(1, 3), padding=1),
            Conv2dBNReLU(ch5x5, ch5x5, kernel_size=(3, 1), padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            Conv2dBNReLU(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class Classifier(nn.Module):

    def __init__(self, in_channels, classes):
        super(Classifier, self).__init__()
        self.in_channels = in_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DOLinear(in_channels, classes)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, self.in_channels)
        x = self.fc(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


class AuxClassifier(nn.Module):

    def __init__(self, in_channels, classes):
        super(AuxClassifier, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = Conv2dBNReLU(in_channels, 1024, kernel_size=1)
        self.fc1 = DOLinearBNReLU(1024, 1024)
        self.fc2 = DOLinear(1024, classes)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


class GoogleNet(nn.Module):

    def __init__(self, in_channels=3, classes=1000, ver=1):
        super(GoogleNet, self).__init__()

        if ver == 1:
            block = InceptionBlockV1
        elif ver == 2:
            block = InceptionBlockV2
        elif ver == 3:
            block = InceptionBlockV3
        else:
            raise ValueError('Version must be 1/2/3')

        self.stem = nn.Sequential(
            Conv2dBNReLU(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            Conv2dBNReLU(64, 64, kernel_size=1),
            Conv2dBNReLU(64, 192, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )

        self.inception_1a = block(192, 64, 96, 128, 16, 32, 32)
        self.inception_1b = block(256, 128, 128, 192, 32, 96, 64)

        self.pool_1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception_2a = block(480, 192, 96, 208, 16, 48, 64)
        self.inception_2b = block(512, 160, 112, 224, 24, 64, 64)
        self.inception_2c = block(512, 128, 128, 256, 24, 64, 64)
        self.inception_2d = block(512, 112, 144, 288, 32, 64, 64)
        self.inception_2e = block(528, 256, 160, 320, 32, 128, 128)

        self.pool_2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception_3a = block(832, 256, 160, 320, 32, 128, 128)
        self.inception_3b = block(832, 384, 192, 384, 48, 128, 128)

        self.classifier = Classifier(1024, classes)

        self.aux_classifier_1 = AuxClassifier(512, classes)

        self.aux_classifier_2 = AuxClassifier(528, classes)

        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_1a(x)
        x = self.inception_1b(x)
        x = self.pool_1(x)
        x = self.inception_2a(x)
        aux_1 = self.aux_classifier_1(x)
        x = self.inception_2b(x)
        x = self.inception_2c(x)
        x = self.inception_2d(x)
        aux_2 = self.aux_classifier_2(x)
        x = self.inception_2e(x)
        x = self.pool_2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.classifier(x)
        # return x, aux_1, aux_2
        return x * 0.333 + aux_1 * 0.333 + aux_2 * 0.333

    def predict(self, x):
        x, aux_1, aux_2 = self.forward(x)
        x = self.sm(x)
        aux_1 = self.sm(aux_1)
        aux_2 = self.sm(aux_2)
        # return x, aux_1, aux_2
        return x * 0.333 + aux_1 * 0.333 + aux_2 * 0.333
