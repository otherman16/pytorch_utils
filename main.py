from pytorch_utils import PyTorchTrainer, Transforms, TrainTransforms, TensorBoardLogger, make_image_label_figure

from vgg import VGG
from mobilenet import MobileNet, MobileNetV2
from alexnet import AlexNet
from resnet import ResNet
from mnistnet import MNISTNet
from googlenet import GoogleNet
from fpn_resnet import fpn_resnet18, fpn_resnet34, fpn_resnet50

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
import torch.optim.lr_scheduler as lr_scheduler


def main():
    origin_channels = 3
    in_channels = 3
    size = (224, 224)
    batch_size = 128
    classes = 20
    class_names = VOC2012_CLASSES
    root = '../../datasets'

    transform_train = TrainTransforms(in_channels=origin_channels, out_channels=in_channels, size=size,
                                      horizontal_flip=True, random_affine=False, random_erasing=False, color_jitter=False)
    transform_val = Transforms(in_channels=origin_channels, out_channels=in_channels, size=size)

    # trainset = MNIST_DATASET(root, train=True, transform=transform_train, download=False)
    # valset = MNIST_DATASET(root, train=False, transform=transform_val, download=False)
    # trainset = FASHION_MNIST_DATASET(root, train=True, transform=transform_train, download=False)
    # valset = FASHION_MNIST_DATASET(root, train=False, transform=transform_val, download=False)
    # trainset = CIFAR10_DATASET(root, train=True, transform=transform_train, download=False)
    # valset = CIFAR10_DATASET(root, train=False, transform=transform_val, download=False)
    trainset = VOC2012_CLASSIFICATION_DATASET(root, train=True, transform=transform_train, download=False)
    valset = VOC2012_CLASSIFICATION_DATASET(root, train=False, transform=transform_val, download=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # net = AlexNet(in_channels=in_channels, classes=classes)s
    # net = VGG(in_channels=in_channels, classes=classes, cfg='A')
    # net = MobileNet(width_multiplier=1.0, in_channels=in_channels, classes=classes)
    # net = MobileNetV2(width_multiplier=1.0, in_channels=in_channels, classes=classes)
    # net = ResNet(in_channels=in_channels, classes=classes, cfg='18')
    # net = MNISTNet(in_channels=in_channels, classes=classes)
    # net = GoogleNet(in_channels=in_channels, classes=classes, ver=2)
    net = fpn_resnet18(classes=classes, pretrained=True, freeze=True)
    # net = fpn_resnet34(classes=classes, pretrained=True)

    print(net)

    net = net.to(torch.device(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2, verbose=False, mode='min',
                                               threshold=0.0001, cooldown=0, min_lr=0.00001)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # os.makedirs(logdir)
    logger = TensorBoardLogger(logdir, class_names=class_names, testloader=testloader, device=device)
    trainer = PyTorchTrainer(device=device, epoch_callback=logger.epoch_callback, batch_callback=logger.batch_callback)
    trainer.train(net, optimizer, criterion, trainloader, valloader, logdir, scheduler=scheduler, epochs=10)


if __name__ == "__main__":
    main()
