import sys
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from utils import config

from discriminator_block import DiscriminatorBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 64

        self.layers = []

        for idx in range(3):
            self.layers.append(
                DiscriminatorBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    use_batch_norm=False if idx == 0 else True,
                )
            )
            self.in_channels = self.out_channels
            self.out_channels *= 2

        self.block = nn.Sequential(*self.layers)

        self.output = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.output(self.block(x))

        else:
            raise Exception(
                "Please provide the tensor for further process".capitalize()
            )

    @staticmethod
    def total_params(model=None):
        if isinstance(model, Discriminator):
            return sum(p.numel() for p in model.parameters())

        else:
            raise Exception("Please provide the model for further process".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Discriminator for DualGAN".title()
    )
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Define the channels".capitalize()
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    config_files = config()

    netD = Discriminator(in_channels=in_channels)

    assert netD(torch.randn(1, 3, 256, 256)).size() == (1, 1, 30, 30)

    summary(model=netD, input_size=(3, 256, 256))

    draw_graph(model=netD, input_data=torch.randn(1, 3, 256, 256)).visual_graph.render(
        filename=os.path.join(config_files["path"]["files_path"], "netD"),
        format=(
            "png"
            if config_files["path"]["files_path"]
            else "Cannot be saved netD file".capitalize()
        ),
    )
