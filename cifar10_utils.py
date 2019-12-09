import torchvision.datasets as datasets


CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def CIFAR10_DATASET(root='./data', train=True, transform=None, target_transform=None, download=False):
    return datasets.CIFAR10(root, train=train, transform=transform, target_transform=target_transform,
                            download=download)

