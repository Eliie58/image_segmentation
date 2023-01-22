"""
U-Net Model Utils
"""
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transform
from torch.utils.data import Dataset


class Convblock(nn.Module):
    """
    Convlution Block
    """

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernal=3,
                 stride=1,
                 padding=1):

        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernal, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernal),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor):
        """
        Step Forward
        """
        out_tensor = self.convblock(tensor)
        return out_tensor


class UNet(nn.Module):
    """
    UNet Neural Network
    """

    def __init__(self, input_channel, retain=True):

        super().__init__()

        self.conv1 = Convblock(input_channel, 32)
        self.conv2 = Convblock(32, 64)
        self.conv3 = Convblock(64, 128)
        self.conv4 = Convblock(128, 256)
        self.neck = nn.Conv2d(256, 512, 3, 1)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 3, 2, 0, 1)
        self.dconv4 = Convblock(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 0, 1)
        self.dconv3 = Convblock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 3, 2, 0, 1)
        self.dconv2 = Convblock(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 3, 2, 0, 1)
        self.dconv1 = Convblock(64, 32)
        self.out = nn.Conv2d(32, 3, 1, 1)
        self.retain = retain

    def forward(self, tensor):
        """
        Encoder Network
        """

        # Conv down 1
        conv1 = self.conv1(tensor)
        pool1 = F.max_pool2d(conv1, kernel_size=2, stride=2)
        # Conv down 2
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2, stride=2)
        # Conv down 3
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2, stride=2)
        # Conv down 4
        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2, stride=2)

        # BottelNeck
        neck = self.neck(pool4)

        # Decoder Network

        # Upconv 1
        upconv4 = self.upconv4(neck)
        croped = crop(conv4, upconv4)
        # Making the skip connection 1
        dconv4 = self.dconv4(torch.cat([upconv4, croped], 1))
        # Upconv 2
        upconv3 = self.upconv3(dconv4)
        croped = crop(conv3, upconv3)
        # Making the skip connection 2
        dconv3 = self.dconv3(torch.cat([upconv3, croped], 1))
        # Upconv 3
        upconv2 = self.upconv2(dconv3)
        croped = crop(conv2, upconv2)
        # Making the skip connection 3
        dconv2 = self.dconv2(torch.cat([upconv2, croped], 1))
        # Upconv 4
        upconv1 = self.upconv1(dconv2)
        croped = crop(conv1, upconv1)
        # Making the skip connection 4
        dconv1 = self.dconv1(torch.cat([upconv1, croped], 1))
        # Output Layer
        out = self.out(dconv1)

        if self.retain:
            out = F.interpolate(out, list(tensor.shape)[2:])

        return out


class MyDataset(Dataset):
    """
    Dataset
    """

    def __init__(self, images_path, transform_img=None, transform_label=None):
        self.images_path = images_path
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = plt.imread(self.images_path[idx])
        image = img[:, : int(img.shape[1] / 2)]
        label = img[:, int(img.shape[1] / 2):]

        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_label:
            label = self.transform_label(label)

        return image, label


def crop(input_tensor, target_tensor):
    """
    For making the size of the encoder conv layer and the
    decoder Conv layer same
    """
    _, _, height, width = target_tensor.shape
    return transform.CenterCrop([height, width])(input_tensor)
