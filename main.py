from pytorch_utils import PyTorchTrainer, Transforms, batch2image

from vgg import VGG
from mobilenet import MobileNet, MobileNetV2
from alexnet import AlexNet
from resnet import ResNet

from mnist_utils import MNIST_CLASSES, MNIST_DATASET
from fashion_mnist_utils import FASHION_MNIST_CLASSES, FASHION_MNIST_DATASET
from cifar10_utils import CIFAR10_CLASSES, CIFAR10_DATASET
from voc_utils import VOC2012_CLASSES, VOC2012_CLASSIFICATION_DATASET, VOC2012_DETECTION_DATASET, VOC2012_SEGMENTATION_DATASET

import os
import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


def main():
    origin_channels = 1
    in_channels = 3
    size = (64, 64)
    batch_size = 256
    classes = 20
    root = '../../datasets'

    transform_train = Transforms(in_channels=origin_channels, out_channels=in_channels, size=size, train=True,
                                 horizontal_flip=False)
    transform_val = Transforms(in_channels=origin_channels, out_channels=in_channels, size=size)

    trainset = MNIST_DATASET(root, train=True, transform=transform_train, download=False)
    valset = MNIST_DATASET(root, train=False, transform=transform_val, download=False)
    # trainset = FASHION_MNIST_DATASET(root, train=True, transform=transform_train, download=False)
    # valset = FASHION_MNIST_DATASET(root, train=False, transform=transform_val, download=False)
    # trainset = CIFAR10_DATASET(root, train=True, transform=transform_train, download=False)
    # valset = CIFAR10_DATASET(root, train=False, transform=transform_val, download=False)
    # trainset = VOC2012_CLASSIFICATION_DATASET(root, train=True, transform=transform_train, download=False)
    # valset = VOC2012_CLASSIFICATION_DATASET(root, train=False, transform=transform_val, download=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # net = AlexNet(in_channels=in_channels, classes=classes)s
    # net = VGG(in_channels=in_channels, classes=classes, cfg='A')
    # net = MobileNet(width_multiplier=1.0, in_channels=in_channels, classes=classes)
    # net = MobileNetV2(width_multiplier=1.0, in_channels=in_channels, classes=classes)
    # net = ResNet(in_channels=in_channels, classes=classes, cfg='18')

    net = torchvision.models.resnet18(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(net.fc.in_features, classes)
    )

    # net = torchvision.models.mobilenet_v2(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    # net.classifier = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(1280, classes)
    # )

    print(net)

    net = net.to(torch.device(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)

    # optimizer = optim.SGD(net.parameters(), lr=0.045, momentum=0.9, weight_decay=0.00004)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    # scheduler = MultiStepLR(optimizer, milestones=[2, 5, 10, 15], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True,
    #                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # os.makedirs(logdir)
    logger = SummaryWriter(logdir)
    img_grid = batch2image(trainloader)
    logger.add_image('Images', img_grid)
    trainer = PyTorchTrainer(logger=logger, path=logdir, device=device)
    trainer.train(net, optimizer, criterion, trainloader, valloader, scheduler=None, num_epochs=20)


if __name__ == "__main__":
    main()
