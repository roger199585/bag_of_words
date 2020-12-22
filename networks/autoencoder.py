"""
Author: Corn
Porgram:
    Autoencoder 模型，要拿來當 feature extractor，這個 feature 拿來做後續的 Anomaly detection 用
History:
    2020/12/17   First release
"""

from torch import nn

class autoencoder(nn.Module):
    def __init__(self, in_channel, min_resolution=4):
        super(autoencoder, self).__init__()
        self.channel_mult = 8
        self.in_channel = in_channel
        self.min_resolution = min_resolution

        self.encoder8 = nn.Sequential(
            # b, 8, 64, 64
            nn.Conv2d(self.in_channel, self.channel_mult, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 16, 32, 32
            nn.Conv2d(self.channel_mult, self.channel_mult * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 32, 16, 16
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 16, 8, 8
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 2, 8, 8
            nn.Conv2d(self.channel_mult * 2, self.channel_mult // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel_mult // 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder8 = nn.Sequential(
            # b, 16, 8, 8
            Interpolate(size=(10, 10), mode='bilinear'),
            nn.Conv2d(self.channel_mult // 4, self.channel_mult * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 32, 16, 16
            Interpolate(size=(18, 18), mode='bilinear'),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 16, 32, 32
            Interpolate(size=(34, 34), mode='bilinear'),
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 8, 64, 64
            Interpolate(size=(66, 66), mode='bilinear'),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 3, 64, 64
            Interpolate(size=(66, 66), mode='bilinear'),
            nn.Conv2d(self.channel_mult, self.in_channel, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.encoder4 = nn.Sequential(
            # b, 8, 64, 64
            nn.Conv2d(self.in_channel, self.channel_mult, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 16, 32, 32
            nn.Conv2d(self.channel_mult, self.channel_mult * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 32, 16, 16
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 16, 8, 8
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 8, 4, 4
            nn.Conv2d(self.channel_mult * 2, self.channel_mult, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder4 = nn.Sequential(
            # b, 16, 8, 8
            Interpolate(size=(10, 10), mode='bilinear'),
            nn.Conv2d(self.channel_mult, self.channel_mult * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 32, 16, 16
            Interpolate(size=(18, 18), mode='bilinear'),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 16, 32, 32
            Interpolate(size=(34, 34), mode='bilinear'),
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 8, 64, 64
            Interpolate(size=(66, 66), mode='bilinear'),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),

            # b, 3, 64, 64
            Interpolate(size=(66, 66), mode='bilinear'),
            nn.Conv2d(self.channel_mult, self.in_channel, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_code = self.encoder8(x) if self.min_resolution == 8 else self.encoder4(x)
        reconstruct_image = self.decoder8(latent_code) if self.min_resolution == 8 else self.decoder4(latent_code)
        return reconstruct_image, latent_code


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x