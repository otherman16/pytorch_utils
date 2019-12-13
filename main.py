from pytorch_utils import PyTorchTrainer, Transforms, TrainTransforms, TensorBoardLogger

from vgg import VGG
from mobilenet import MobileNet, MobileNetV2
from alexnet import AlexNet
from resnet import ResNet
from mnistnet import MNISTNet

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
    origin_channels = 1
    in_channels = 1
    size = (28, 28)
    batch_size = 512
    classes = 10
    root = '../../datasets'

    transform_train = TrainTransforms(in_channels=origin_channels, out_channels=in_channels, size=size,
                                      horizontal_flip=False, random_affine=True, random_erasing=True)
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
    net = MNISTNet(in_channels=in_channels, classes=classes)

    # net = torchvision.models.resnet18(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    # net.fc = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(net.fc.in_features, classes)
    # )

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
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2, verbose=False, mode='min',
                                               threshold=0.0001, cooldown=0, min_lr=0.00001)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(logdir)
    # logger = TensorBoardLogger(logdir, class_names=MNIST_CLASSES)
    trainer = PyTorchTrainer(device=device)#, epoch_callback=logger.epoch_callback, batch_callback=logger.batch_callback)
    trainer.train(net, optimizer, criterion, trainloader, valloader, logdir, scheduler=scheduler, epochs=10)

    # net.eval()
    # for param in net.parameters():
    #     param.requires_grad = False
    #
    # bad_images = []
    # bad_predictions = []
    # for images, targets in valloader:
    #     images = images.to(torch.device(device))
    #     targets = targets.to(torch.device(device))
    #     outputs = net.forward(images)
    #     predictions = outputs.argmax(dim=1).data
    #     for image, target, prediction in zip(images, targets, predictions):
    #         if target != prediction:
    #             bad_images.append(image.cpu())
    #             bad_predictions.append(prediction.cpu())
    #
    # bad_images = torch.stack(bad_images)
    # fig = make_image_label_figure(bad_images, bad_predictions, class_names=MNIST_CLASSES)
    # logger.add_figure('Bad predictions', fig)
    #
    # while True:
    #     print('done')
    #     k = input()
    #     if k == 'q':
    #         break


if __name__ == "__main__":
    main()
