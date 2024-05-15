from CVD_Simulation import *
from contrast import *
from database import *
from engine import *

import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from kornia.color import rgb_to_lab
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torchvision.utils import save_image


class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        """Computes the structural similarity (SSIM) index map between two images.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel. Default: (11, 11)
        sigma : float
            Standard deviation of the gaussian kernel in the x and y direction.
        """
        super(SSIMLoss, self).__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float):
        """Function to create gaussian kernel of size (kernel_size, kernel_size, 3), for 3 channels"""
        # Generating mean-centered data, i.e. mean = 0
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

        # Creating 1D gaussian distributed values
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))   # Count exp power of normal dist given that mean = 0
        gauss = (gauss / gauss.sum()).unsqueeze(dim=0)      # normalize data (нормируем данные, чтобы в сумме получали 1, i.e. интеграл = 1)

        # Form 2D gaussian distributed values
        gauss2D = torch.matmul(gauss.t(), gauss)  # form kernel_size rows, kernel_size cols of normalized data, distributed normally
        gauss2Dx3 = gauss2D.expand(3, 1, kernel_size, kernel_size).contiguous() # view (not reshape, not changing an object) 2D normally distributed tensor as 3 times of 2D tensor (for 3 image channels)

        return torch.Tensor(gauss2Dx3)

    def forward(self, x: torch.Tensor, y: torch.Tensor, as_loss: bool = True) -> torch.Tensor:
        # Check that generated self.gaussian_kernel is stored in the same GPU device as first image
        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        # Get SSIM map between two images
        ssim_map = self._ssim(x, y)

        if as_loss:
            # get naturalness loss (SSIM is in range [-1, 1], 1 - perfect match (similarity), -1 - not)
            return 1 - ssim_map.mean()
        else:
            # otherwise, get SSIM map index
            return ssim_map

    # See https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    def _ssim(self, x: torch.Tensor, y: torch.Tensor):
        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)    # calculate M[X**2]
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)    # calculate M[Y**2]
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)    # calculate M[X*Y] (for covariance)
        vx = uxx - ux * ux  # DX := M[X**2] - (MX)**2
        vy = uyy - uy * uy  # DY := M[Y**2] - (MY)**2
        vxy = uxy - ux * uy # cov(X, Y) := M[X*Y] - M[X]*M[Y]

        # To avoid zero division - add c1 and c2
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        return torch.Tensor( (( (2*ux*uy + c1) * (2*vxy + c2) ) / ( (ux*ux + uy*uy + c1) * (vx + vy + c2) + 1e-12 )) )


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs.type(Tensor))
    fake_B = generator(real_A)
    real_A = transforms_1(real_A)
    fake_B = transforms_1(fake_B)

    cvd_fake = CVDSimulation(fake_B, CVD_type, 100)
    cvd_original = CVDSimulation(real_A, CVD_type, 100)

    img_sample_1 = torch.cat((real_A.data, fake_B.data), -2)
    img_sample_2 = torch.cat((cvd_original.data, cvd_fake.data), -2)
    img_sample = torch.cat((img_sample_1.data, img_sample_2.data), -1)

    save_image(img_sample, "images/ssim/%s/%s.png" % (dataset_name + suffix[CVD_type.value], batches_done), nrow=5,
                normalize=True)


# ------------------------------------------
# ----------SOME GLOBAL CONSTANTS-----------
# ------------------------------------------
cuda = True if torch.cuda.is_available() else False

n_epochs = 121  # Number of epochs
batch_size = 8
lr = 0.0002 # Adam learning rate
b1 = 0.5    # Adam b1
b2 = 0.999  # Adam b2
img_height = 256
img_width = 256
sample_interval = 200   # sample images after 200 iterations
checkpoint_interval = 5 # make tensorboard model checkpoints after each 5 epochs
lambda_ssim = 0.00      # weight of SSIM-loss count metric
global_points = 3000    # number of points to randomly take and measure contrast on

dataset_name = "CVD"
CVD_type = CVDType.DEUTAN    # 0 = Deutan, 1 = Protan, 2 = Tritan


# making folders for sample images and model checkpoints
suffix = [
    '_DEUTAN_lab_globalContrast_contrastPoints%d_ssim_weigth_%s' % (global_points, lambda_ssim),   # Deutan
    '_PROTAN_lab_globalContrast_contrastPoints%d_ssim_weigth_%s' % (global_points, lambda_ssim),   # Protan
    '_TRITAN_lab_globalContrast_contrastPoints%d_ssim_weigth_%s' % (global_points, lambda_ssim),   # Tritan
]

os.makedirs("images/ssim/%s" % dataset_name + suffix[CVD_type.value], exist_ok = True)
os.makedirs("saved_models/%s" % dataset_name + suffix[CVD_type.value], exist_ok = True)


# Setup neural nets
criterion_contrast = torch.nn.L1Loss()  # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
generator = SWIN_Generator()
if cuda:
    generator = generator.cuda()
    criterion_contrast.cuda()
# summary(generator, input_size=(3, 256, 256))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))


# Dataloaders preparation
data_path = "./plates"

transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(data_path, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

val_dataloader = DataLoader(
    ImageDataset(data_path, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

transforms_1 = transforms.Compose([transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))])
transforms_2 = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms_3 = transforms.Compose([transforms.Normalize((0, 0, 0), (100, 128, 128))])
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# TRAINING
prev_time = time.time()
writer = SummaryWriter("tensorboard/%s" % dataset_name + suffix[CVD_type.value], flush_secs=30)
ssimloss_funtion = SSIMLoss(kernel_size=11)

G_Loss = []
for epoch in range(n_epochs):
    G_losses = []       # Generator (SWIN) losses
    local_loss = []     # local loss contrast (L1 metrics) on original and generated (by SWIN Generator) images
    global_loss = []    # global loss contrast (L1 metrics) on original and generated (by SWIN Generator) images
    ssim_loss = []      # ssim loss in LAB space between original and generated images

    for i, batch in enumerate(dataloader):
        real_A = Variable(batch.type(Tensor))   # differentiating by it

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # Generate fake iamge
        fake_B = generator(real_A)

        real_A = transforms_1(real_A)
        fake_B = transforms_1(fake_B)

        # Calculate losses (with severity = 100)
        cvd_fake = CVDSimulation(fake_B, CVD_type, 100)
        cvd_original = CVDSimulation(real_A, CVD_type, 100)

        # Convert to lab spaces
        cvd_fake = rgb_to_lab(cvd_fake)
        cvd_original = rgb_to_lab(cvd_original)
        real_A = rgb_to_lab(real_A)

        fake_B = transforms_3(rgb_to_lab(fake_B))
        cvd_fake = transforms_3(cvd_fake)
        cvd_original = transforms_3(cvd_original)
        real_A = transforms_3(real_A)

        # Count contrasts for initial Image, generated one and both of them in terms of LAB space
        contrast_1 = calculateLocalContrastL1(real_A, window_size=5)
        contrast_2 = calculateLocalContrastL1(cvd_fake, window_size=5)
        g_contrast_1, g_contrast_2 = calculateGlobalContrastL1(real_A, cvd_fake, global_points)

        # Count losses between generated and initial images (L1 metrics)
        loss_contrast = criterion_contrast(contrast_1, contrast_2)
        loss_nature = criterion_contrast(real_A, fake_B)
        loss_contrast_global = criterion_contrast(g_contrast_1, g_contrast_2)

        # Count the 'similarity' index between initial and generated images 
        loss_ssim = ssimloss_funtion(transforms_2(real_A), transforms_2(fake_B))

        # Count global loss, i.e. SSIM_WEIGHT * LOSS_{ssim} + (LOSS_{contrast_local} + LOSS_{contrast_global}) * (1 - SSIM_WEIGHT)
        # like linear interpolation
        loss_G = lambda_ssim * loss_ssim + (loss_contrast + loss_contrast_global) * (1 - lambda_ssim)

        # save losses
        local_loss.append(loss_contrast.item())
        global_loss.append(loss_contrast_global.item())
        ssim_loss.append(loss_ssim.item())

        # count gradients in terms of GLOBAL loss
        loss_G.backward()

        # do Adam step to optimize loss_g
        optimizer_G.step()


        # ------------------------------------------
        # -----------------LOGGING------------------
        # ------------------------------------------

        # Determine ETA, i.e. estimated time of arrival
        batches_done = epoch * len(dataloader) + i
        batches_left = n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        G_losses.append(loss_G.item())
        g_l = np.array(G_losses).mean()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d]  [G: %f,  local_c:%f, global_c:%f,ssim:%f] ETA: %s"
            % (
                epoch,
                n_epochs,
                i,
                len(dataloader),
                loss_G.item(),
                loss_contrast.item(),
                loss_contrast_global.item(),
                loss_ssim.item(),
                time_left,
            )
        )

        if batches_done % sample_interval == 0:
            sample_images(batches_done)

    G_Loss.append(g_l)

    writer.add_scalars('loss/G_D_loss', {"G_loss": g_l,
                                         }, epoch)
    writer.add_scalars('loss/Generator_loss', {"loss_contrast": loss_contrast.item(),
                                               "loss_global": loss_contrast_global.item(),
                                               "loss_ssim": loss_ssim.item()}, epoch)


    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(),
                   "saved_models/%s/generator_%d.pth" % (dataset_name + suffix[CVD_type.value], epoch))


# Saving trained model
torch.save(generator.state_dict(), f"./saved_models/{dataset_name + suffix[CVD_type.value]}/[FINAL] generator_{epoch}.pth")