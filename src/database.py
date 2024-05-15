import glob

import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):    # , mode="train", flip_flag=1
        self.transform = transforms.Compose(transforms_)    # torchvision image transformations (sequence)
        self.files = sorted(glob.glob(root + "/*.*"))   # dataset folder (image files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")    # Open image at index position in dataset
        img = self.transform(img)   # apply image transformations

        return img
