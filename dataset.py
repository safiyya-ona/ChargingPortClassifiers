import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, to_tensor
import os
import pandas as pd
from PIL import Image

SIZE = [32, 32]


def uniform_image_size(image: torch.Tensor):
    image = resize(image, SIZE)
    return image


def target_transform(label):
    if label == "type2":
        return torch.tensor([0., 1.])
    else:
        return torch.tensor([1., 0.])


class ConnectorsTraining(Dataset):
    def __init__(self, annotations_file, img_dir="training", transform=uniform_image_size, target_transform=target_transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        image = to_tensor(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class ConnectorsTesting(Dataset):
    def __init__(self, annotations_file, img_dir="testing", transform=uniform_image_size, target_transform=target_transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        image = to_tensor(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    training = ConnectorsTraining("training.csv")
    testing = ConnectorsTesting("testing.csv")
    for i in range(len(testing)):
        testing[i][0].shape
