import sys
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path.append("src/")


class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.decoder = self.block()

    def block(self):
        layers = OrderedDict()

        layers["deconv"] = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
            )
        )

        layers["relu"] = nn.ReLU(inplace=True)
        layers["batch_norm"] = nn.BatchNorm2d(num_features=self.out_channels)

        return nn.Sequential(layers)

    def forward(self, x, skip_info):
        if isinstance(x, torch.Tensor):
            x = self.decoder(x)
            return torch.concat((x, skip_info), dim=1)

        else:
            raise Exception(
                "Please provide the tensor for further process".capitalize()
            )

    @staticmethod
    def total_params(model):
        if isinstance(model, Decoder):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        else:
            raise Exception("Please provide the model for further process".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder block for netG".title())
    parser.add_argument("--in_channels", type=int, default=512)
    parser.add_argument("--out_channels", type=int, default=512)

    args = parser.parse_args()

    decoder = Decoder(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
