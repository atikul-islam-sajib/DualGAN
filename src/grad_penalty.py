import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from discriminator import Discriminator


class GradientPenalty(nn.Module):
    def __init__(self, in_channels=3, batch_size=1, device="mps"):
        super(GradientPenalty, self).__init__()

        self.in_channels = in_channels
        self.batch_size = batch_size
        self.device = device

        self.alpha = torch.randn(self.batch_size, self.in_channels, 1, 1).to(
            self.device
        )

    def forward(self, netD, X, y):
        if (
            isinstance(netD, Discriminator)
            and isinstance(X, torch.Tensor)
            and isinstance(y, torch.Tensor)
        ):
            interpolated = (self.alpha * X) + ((1 - self.alpha) * y)
            interpolated = interpolated.requires_grad_(True)
            d_interpolated = netD(interpolated)

            gradients = torch.autograd.grad(
                outputs=d_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones(d_interpolated.size()).to(self.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            gradients = gradients.view(gradients.size(0), -1)
            gradients = torch.norm(gradients, 2, dim=1)
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
    device = args.device

    netD = Discriminator(in_channels=in_channels)

    X = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)

    grad_penalty = GradientPenalty(
        in_channels=in_channels, batch_size=batch_size, device=device
    )

    print(grad_penalty(netD, X, y))
