# internal imports
import math
from collections import OrderedDict
from typing import Callable, Optional, Union

# external imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudIdentity(nn.Module):
    def __init__(self, activation: Callable = torch.tanh, squash: bool = True):
        super().__init__()
        self.activation = activation
        self.squash = squash

    def forward(self, cloudy_input, cloud_pred):
        cloud_pred = self.activation(cloud_pred)
        if self.squash:
            cloud_pred = cloud_pred * 0.5 - 0.5
        return cloud_pred


class CloudAddition(CloudIdentity):
    def __init__(self, activation: Callable = torch.tanh, squash: bool = True):
        super().__init__()
        self.activation = activation
        self.squash = squash

    def forward(self, cloudy_input, cloud_pred):
        cloud_pred = self.activation(cloud_pred)
        if self.squash:
            cloud_pred = cloud_pred * 0.5 - 0.5
        return cloudy_input + cloud_pred


class CloudDivision(CloudIdentity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, cloudy_input, cloud_pred):
        if self.activation is not None:
            cloud_pred = self.activation(cloud_pred)
        if self.squash:
            cloud_pred = cloud_pred * 0.5 + 0.5
        outputs = cloudy_input / (cloud_pred + 1e-5)
        return outputs


class TrueUNet(nn.Module):
    def __init__(self, n_blocks: int = 4, cleaner=CloudSubtraction, in_channels=1, out_channels=1, init_features=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.cleaner = cleaner()
        self.features = init_features

        self.encoder1 = self._block(in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._block(self.features * 8, self.features * 16, name="bottleneck")
        self.upconv4 = nn.ConvTranspose2d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = self._block(self.features * 2, self.features, name="dec1")
        self.conv = nn.Conv2d(in_channels=self.features, out_channels=out_channels, kernel_size=1)

        if self.n_blocks >= 5:
            self.encoder5 = self._block(self.features * 8, self.features * 16, name="enc4")
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.upconv5 = nn.ConvTranspose2d(self.features * 32, self.features * 16, kernel_size=2, stride=2)
            self.decoder5 = self._block((self.features * 16) * 2, self.features * 16, name="dec4")
            self.bottleneck = self._block(self.features * 16, self.features * 32, name="bottleneck")

        if self.n_blocks >= 6:
            self.encoder6 = self._block(self.features * 16, self.features * 32, name="enc4")
            self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.upconv6 = nn.ConvTranspose2d(self.features * 64, self.features * 32, kernel_size=2, stride=2)
            self.decoder6 = self._block((self.features * 32) * 2, self.features * 32, name="dec4")
            self.bottleneck = self._block(self.features * 32, self.features * 64, name="bottleneck")
        if self.n_blocks == 7:
            self.encoder7 = self._block(self.features * 32, self.features * 64, name="enc7")
            self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.upconv7 = nn.ConvTranspose2d(self.features * 128, self.features * 64, kernel_size=2, stride=2)
            self.decoder7 = self._block((self.features * 64) * 2, self.features * 64, name="dec7")
            self.bottleneck = self._block(self.features * 64, self.features * 128, name="bottleneck")

    def forward_4(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    def forward_5(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    def forward_6(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))
        bottleneck = self.bottleneck(self.pool6(enc6))
        dec6 = self.upconv6(bottleneck)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.decoder6(dec6)
        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    def forward_7(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))
        enc7 = self.encoder7(self.pool6(enc6))

        bottleneck = self.bottleneck(self.pool7(enc7))

        dec7 = self.upconv7(bottleneck)
        dec7 = torch.cat((dec7, enc7), dim=1)
        dec7 = self.decoder7(dec7)

        dec6 = self.upconv6(dec7)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.decoder6(dec6)
        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)


    def forward(self, x):
        if self.n_blocks == 4:
            out = self.forward_4(x)
        elif self.n_blocks == 5:
            out = self.forward_5(x)
        elif self.n_blocks == 6:
            out = self.forward_6(x)
        elif self.n_blocks == 7:
            out = self.forward_7(x)
        else:
            raise ValueError(f"Incorrect number of blocks used: {self.n_blocks}")
        return self.cleaner(x, out)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def downsample(
        in_filters,
        out_filters,
        size,
        apply_batchnorm=True,
        normal_init_weights: bool = True,
):
    modules: list[nn.Module] = [
        Conv2dSame(in_filters, out_filters, size, stride=2, bias=False)]
    if normal_init_weights:
        torch.nn.init.normal_(modules[0].weight, 0., 0.02)
    if apply_batchnorm:
        modules.append(nn.BatchNorm2d(out_filters))
    modules.append(nn.LeakyReLU(inplace=True))
    return nn.Sequential(*modules)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_stack = nn.Sequential(
            downsample(2, 64, 4, False),
            downsample(64, 128, 4),
            downsample(128, 256, 4),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, stride=1))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0., 0.02)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.down_stack(x)


class Pix2Pix(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_fn(disc_real_output, torch.ones_like(disc_real_output))
        generated_loss = self.loss_fn(disc_generated_output, torch.zeros_like(disc_generated_output))
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target, lamb=100):
        """
        Computes the Generator's loss
        Params :
        --------
        disc_generated_output (tensor) : output of the dicriminator
        gen_output (tensor) : output of the generator
        target (int) : target image
        lamb (float) : weight coeff
        Returns :
        ---------
        total_gen_loss (float) : generator loss
        gan_loss (float) : gan loss
        l1_loss (float) : mean absolute error for pixel-wise error
        """
        gan_loss = self.loss_fn(torch.ones_like(disc_generated_output), disc_generated_output)
        diff = torch.abs(target - gen_output)
        l1_loss = torch.mean(diff)  # pixel-wise error
        total_gen_loss = gan_loss + (lamb * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def forward(self, inputs):
        return self.generator(inputs)
