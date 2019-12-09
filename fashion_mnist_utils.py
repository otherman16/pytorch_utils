import torchvision.datasets as datasets


FASHION_MNIST_CLASSES = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                         'Ankle Boot')


def FASHION_MNIST_DATASET(root='./data', train=True, transform=None, target_transform=None, download=False):
    return datasets.FashionMNIST(root, train=train, transform=transform, target_transform=target_transform,
                                 download=download)
