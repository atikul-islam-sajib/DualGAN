import sys
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, use_batch_norm=False):
        super(DiscriminatorBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = use_batch_norm

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.discriminator = self.block()

    def block(self):
        layers = OrderedDict()

        layers["conv"] = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
            )
        )

        if self.batch_norm:
            layers["batch_norm"] = nn.BatchNorm2d(num_features=self.out_channels)

        layers["leaky_ReLU"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.discriminator(x)

        else:
            raise Exception(
                "Please provide the tensor for further process".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discriminator for the DiscoGAN".capitalize()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Number of output channels".capitalize(),
    )
    args = parser.parse_args()

    netD_block = DiscriminatorBlock(
        in_channels=args.in_channels, out_channels=args.out_channels
    )
