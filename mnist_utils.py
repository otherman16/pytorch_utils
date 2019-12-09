import torchvision.datasets as datasets


MNIST_CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


def MNIST_DATASET(root='./data', train=True, transform=None, target_transform=None, download=False):
    return datasets.MNIST(root, train=train, transform=transform, target_transform=target_transform,
                          download=download)
