import sys
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path.append("src/")


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, use_batch_norm=True):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = use_batch_norm

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.encoder = self.block()

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

        layers["leaky_relu"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if self.batch_norm:
            layers["batch_norm"] = nn.BatchNorm2d(num_features=self.out_channels)

        return nn.Sequential(layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.encoder(x)

        else:
            raise Exception(
                "Please provide the tensor for further process".capitalize()
            )

    @staticmethod
    def total_params(model):
        if isinstance(model, Encoder):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        else:
            raise Exception("Please provide the model for further process".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--use_batch_norm", type=bool, default=True)
    args = parser.parse_args()

    encoder = Encoder(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        use_batch_norm=args.use_batch_norm,
    )

    print(encoder)
    print(encoder.total_params(encoder))
    print(encoder(torch.randn(1, 3, 256, 256)).shape)
