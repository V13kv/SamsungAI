from CVD_Simulation import *
from engine import SWIN_Generator

import sys
import os

import torch

from torchvision.transforms import v2
from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image
import numpy as np


class UnNormalize(object):
    """Denormalize image with specified mean and std"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m) # The normalize code -> t.sub_(m).div_(s)

        return tensor


def hint():
    print("""python recolor.py <abs_image_path> <cvd_type> <output_path>
abs_image_path - absolute image path (image to recolor).
cvd_type - type of colorblindness:
          0 - Deutan
          1 - Protan
          2 - Tritan
output_path - where to output recolored image (absolute path)
""")


if __name__ == '__main__':
    if len(sys.argv) < 4 or \
       not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[3]) or not str(sys.argv[2]).isdigit() or \
       not 0 <= int(sys.argv[2]) <= 2:
        hint()
        exit()

    # Open image
    imagePath = sys.argv[1]
    cvd_type = int(sys.argv[2])
    outputPath = sys.argv[3]
    img = Image.open(imagePath)

    # transform image
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    transforms_ = v2.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = transforms_(img).type(Tensor)

    # Model load
    model = SWIN_Generator()
    device = torch.device('cuda' if cuda else 'cpu')
    if cvd_type == CVDType.DEUTAN.value:
        if cuda:
            model.load_state_dict(torch.load("../models/DEUTAN/generator_90_epochs.pth"))
        else:
            model.load_state_dict(torch.load("../models/DEUTAN/generator_90_epochs.pth", map_location=device))
    elif cvd_type == CVDType.PROTAN.value:
        if cuda:
            model.load_state_dict(torch.load("../models/PROTAN/PROTAN. 100% SEVERITY. generator_10.pth"))
        else:
            model.load_state_dict(torch.load("../models/DEUTAN/PROTAN. 100% SEVERITY. generator_10.pth", map_location=device))
    elif cvd_type == CVDType.TRITAN.value:
        if cuda:
            model.load_state_dict(torch.load("../models/TRITAN/TRITAN. 100% SEVERITY. generator_10.pth"))
        else:
            model.load_state_dict(torch.load("../models/TRITAN/TRITAN. 100% SEVERITY. generator_10.pth", map_location=device))
    model.eval()

    # Evaluation
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    with torch.no_grad():
        output = model(img.unsqueeze(0))
    output = unorm(output)

    # Output to different file
    imgName = os.path.splitext(os.path.basename(imagePath))[0]
    save_image(output, os.path.join(outputPath, "corrected_" + imgName + ".png"))
    