import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import device_init
from discriminator import Discriminator


class GradientPenalty(nn.Module):
    def __init__(self, in_channels=3, batch_size=1, device="mps"):
        super(GradientPenalty, self).__init__()

        self.in_channels = in_channels
        self.batch_size = batch_size
        self.device = device

    def forward(self, netD, X, y, device=None):
        if (
            isinstance(netD, Discriminator)
            and isinstance(X, torch.Tensor)
            and isinstance(y, torch.Tensor)
        ):
            alpha = torch.randn(
                X.size(0),
                self.in_channels,
                512,
                512,
            ).to(device)

            interpolated = (alpha * X) + ((1 - alpha) * y)
            interpolated = interpolated.requires_grad_(True)

            d_interpolated = netD(interpolated)

            gradients = torch.autograd.grad(
                outputs=d_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones_like(d_interpolated, requires_grad=False).to(
                    device
                ),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradients = gradients.view(gradients.size(0), -1)
            gradients = gradients.norm(2, dim=1)

            return ((gradients - 1) ** 2).mean()

        else:
            raise TypeError("Inputs must be torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient penalty for DualGAN".title())
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Define the channels".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Define the batch size".capitalize()
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Define the device".capitalize()
    )

    args = parser.parse_args()

    in_channels = args.in_channels
    batch_size = args.batch_size
    device = device_init(device=args.device)

    netD = Discriminator(in_channels=in_channels)

    netD = netD.to(device)

    X = torch.randn(4, 3, 256, 256).to(device)
    y = torch.randn(4, 3, 256, 256).to(device)

    grad_penalty = GradientPenalty(
        in_channels=in_channels, batch_size=batch_size, device=device
    )

    print(grad_penalty(netD, X, y, device=device))
