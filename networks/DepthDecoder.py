import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def upsample(img):
    return F.interpolate(img, scale_factor=2, mode="bilinear")


class upsampling_block(nn.Module):
    def __init__(self, nIn, nOut) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(nIn, nOut, 3, padding=1),
            nn.ELU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nOut, nOut, 3, padding=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = upsample(x)
        x = self.conv2(x)
        return x


class prediction_head(nn.Module):
    def __init__(self, nIn, nOut) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nOut, 3, padding=1),
            nn.ELU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv(x)
        x = upsample(x)
        x = self.sigmoid(x)
        return x



class depth_decoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=1) -> None:
        #encoder_features: [1/4 dim, 1/8 dim, 1/16 dim]
        super().__init__()
        self.num_ch_enc = np.array(num_ch_enc)
        num_ch_dec = num_ch_enc[0] // 2

        self.up0 = upsampling_block(num_ch_enc[2], num_ch_enc[1])
        self.up1 = upsampling_block(num_ch_enc[1], num_ch_enc[0])
        self.up2 = upsampling_block(num_ch_enc[0], num_ch_dec)

        self.pre_head0 = prediction_head(num_ch_enc[1], num_output_channels)
        self.pre_head1 = prediction_head(num_ch_enc[0], num_output_channels)
        self.pre_head2 = prediction_head(num_ch_dec, num_output_channels)


    def forward(self, encoder_features): #encoder_features: [1/4 feature, 1/8 feature, 1/16 feature]
        up0 = self.up0(encoder_features[2]) + encoder_features[1]
        up1 = self.up1(encoder_features[1]) + encoder_features[0]
        up2 = self.up2(up1)

        output = [self.pre_head0(up0), self.pre_head1(up1), self.pre_head2(up2)]
        return output #output: [1/4 depth map, 1/2 depth map, full res depth map]