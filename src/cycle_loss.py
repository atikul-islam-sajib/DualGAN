import sys
import os
import argparse
import torch
import torch.nn as nn


class CycleLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CycleLoss, self).__init__()

        self.name = "CycleLoss".title()
        self.reduction = reduction

        self.loss = nn.L1Loss(reduction=self.reduction)

    def forward(self, actual, pred):
        if isinstance(actual, torch.Tensor) and isinstance(pred, torch.Tensor):
            return self.loss(actual, pred)
        else:
            raise TypeError("Inputs must be torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a loss function for CycleGAN".capitalize()
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        choices=["mean", "sum", "none"],
    )

    args = parser.parse_args()

    loss = CycleLoss(reduction=args.reduction)

    actual = torch.tensor([1.0, 0.0, 1.0, 0.0])
    predicted = torch.tensor([1.0, 0.0, 1.0, 1.0])

    print("Total loss {}".format(loss(actual, predicted)))
