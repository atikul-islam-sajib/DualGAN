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
from encoder import Encoder
from decoder import Decoder


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 64
        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.encoder1 = Encoder(
            in_channels=self.in_channels, out_channels=self.out_channels
        )
        self.encoder2 = Encoder(
            in_channels=self.out_channels, out_channels=self.out_channels * 2
        )
        self.encoder3 = Encoder(
            in_channels=self.out_channels * 2, out_channels=self.out_channels * 4
        )
        self.encoder4 = Encoder(
            in_channels=self.out_channels * 4, out_channels=self.out_channels * 8
        )
        self.encoder5 = Encoder(
            in_channels=self.out_channels * 8, out_channels=self.out_channels * 8
        )
        self.encoder6 = Encoder(
            in_channels=self.out_channels * 8, out_channels=self.out_channels * 8
        )
        self.encoder7 = Encoder(
            in_channels=self.out_channels * 8,
            out_channels=self.out_channels * 8,
            use_batch_norm=False,
        )

        self.decoder1 = Decoder(
            in_channels=self.out_channels * 8, out_channels=self.out_channels * 8
        )
        self.decoder2 = Decoder(
            in_channels=self.out_channels * 8 * 2, out_channels=self.out_channels * 8
        )
        self.decoder3 = Decoder(
            in_channels=self.out_channels * 8 * 2, out_channels=self.out_channels * 8
        )
        self.decoder4 = Decoder(
            in_channels=self.out_channels * 8 * 2, out_channels=self.out_channels * 4
        )
        self.decoder5 = Decoder(
            in_channels=self.out_channels * 4 * 2, out_channels=self.out_channels * 2
        )
        self.decoder6 = Decoder(
            in_channels=self.out_channels * 2 * 2,
            out_channels=self.out_channels,
        )

        self.output = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            encoder1 = self.encoder1(x)
            encoder2 = self.encoder2(encoder1)
            encoder3 = self.encoder3(encoder2)
            encoder4 = self.encoder4(encoder3)
            encoder5 = self.encoder5(encoder4)
            encoder6 = self.encoder6(encoder5)
            encoder7 = self.encoder7(encoder6)

            decoder1 = self.decoder1(encoder7, encoder6)
            decoder2 = self.decoder2(decoder1, encoder5)
            decoder3 = self.decoder3(decoder2, encoder4)
            decoder4 = self.decoder4(decoder3, encoder3)
            decoder5 = self.decoder5(decoder4, encoder2)
            decoder6 = self.decoder6(decoder5, encoder1)
            output = self.output(decoder6)

            return output

        else:
            raise Exception(
                "Please provide the tensor for further process".capitalize()
            )

    @staticmethod
    def total_params(model):
        if isinstance(model, Generator):
            return sum(p.numel() for p in model.parameters())

        else:
            raise Exception(
                "Please provide the tensor for further process".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generator for the DiscoGAN".capitalize()
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
        default=3,
        help="Number of output channels".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    netG = Generator(in_channels=in_channels)

    print(summary(model=netG, input_size=(3, 256, 256)))

    config_files = config()
    files_path = config_files["path"]["files_path"]

    draw_graph(model=netG, input_data=torch.randn(1, 3, 256, 256)).visual_graph.render(
        filename=os.path.join(files_path, "netG"),
        format=(
            "png"
            if os.path.exists(files_path)
            else "Cannot saved the netG model architecture".capitalize()
        ),
    )

    assert netG(torch.randn(1, 3, 256, 256)).size() == (1, 3, 256, 256)
