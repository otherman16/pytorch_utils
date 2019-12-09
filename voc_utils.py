import os
import sys
import collections
import torch
import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
from torchvision.datasets.voc import DATASET_YEAR_DICT, download_extract
from PIL import Image
from torchvision.datasets.utils import verify_str_arg

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VOC2012_CLASSES = [
    'person',
    'bird',
    'cat',
    'cow',
    'dog',
    'horse',
    'sheep',
    'aeroplane',
    'bicycle',
    'boat',
    'bus',
    'car',
    'motorbike',
    'train',
    'bottle',
    'chair',
    'diningtable',
    'pottedplant',
    'sofa',
    'tvmonitor'
]


def voc_classification_transforms(img, target, object_num):
    objs = target['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]
    obj = objs[object_num-1 if object_num > 0 else object_num]
    transformed_target = VOC2012_CLASSES.index(obj['name'])
    xmin = int(obj['bndbox']['xmin'])
    ymin = int(obj['bndbox']['ymin'])
    xmax = int(obj['bndbox']['xmax'])
    ymax = int(obj['bndbox']['ymax'])
    transformed_img = img.crop((xmin, ymin, xmax, ymax))
    return transformed_img, transformed_target


def voc_detection_target_transform(target):
    objs = target['annotation']['object']
    width = float(target['annotation']['size']['width'])
    height = float(target['annotation']['size']['height'])
    if not isinstance(objs, list):
        objs = [objs]
    transformed_target = torch.stack([torch.tensor([VOC2012_CLASSES.index(obj['name']),
                                                    float(obj['bndbox']['xmin'])/width,
                                                    float(obj['bndbox']['ymin'])/height,
                                                    float(obj['bndbox']['xmax'])/width,
                                                    float(obj['bndbox']['ymax'])/height])
                                      for obj in objs])
    return transformed_target


def voc_collate_fn(_batch):
    samples = torch.stack([el[0] for el in _batch])
    targets = torch.cat([torch.cat([torch.ones((el[1].shape[0], 1))*idx, el[1]], dim=1)
                         for idx, el in enumerate(_batch)])
    batch = (samples, targets)
    return batch


class VOCClassification(VisionDataset):

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCClassification, self).__init__(root, transforms, transform, target_transform)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.image_set = verify_str_arg(image_set, "image_set",
                                        ("train", "trainval", "val"))

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_fs = [os.path.join(splits_dir, c + '_' + image_set.rstrip('\n') + '.txt') for c in VOC2012_CLASSES]

        file_names = []
        self.object_nums = []
        for split_f in split_fs:
            with open(os.path.join(split_f), "r") as f:
                for line in f.readlines():
                    file_name, object_num = line.strip().split()
                    file_names.append(file_name)
                    self.object_nums.append(int(object_num))

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        if image_set == 'val':
            self.images = self.images[::6]
            self.annotations = self.annotations[::6]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        object_num = self.object_nums[index]

        img, target = voc_classification_transforms(img, target, object_num)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def VOC2012_CLASSIFICATION_DATASET(root='./data', train=True, transform=None, target_transform=None, download=False):
    return VOCClassification(root, image_set='train' if train else 'val', transform=transform,
                             target_transform=target_transform, download=download)


def VOC2012_DETECTION_DATASET(root='./data', train=True, transform=None, target_transform=None, download=False):
    return datasets.VOCDetection(root, image_set='train' if train else 'val', transform=transform,
                                 target_transform=target_transform, download=download)


def VOC2012_SEGMENTATION_DATASET(root='./data', train=True, transform=None, target_transform=None, download=False):
    return datasets.VOCSegmentation(root, image_set='train' if train else 'val', transform=transform,
                                    target_transform=target_transform, download=download)
