import sys
import os
import torch
import torch.optim as optim


sys.path.append("src/")

from utils import load, config
from generator import Generator
from discriminator import Discriminator
from grad_penalty import GradientPenalty
from cycle_loss import CycleLoss


def load_dataloader():
    config_files = config()
    dataloader_path = config_files["path"]["processed_path"]

    if os.path.exists(dataloader_path):
        train_dataloader = load(
            filename=os.path.join(dataloader_path, "train_dataloader.pkl")
        )
        test_dataloader = load(
            filename=os.path.join(dataloader_path, "test_dataloader.pkl")
        )

        return {
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
        }

    else:
        raise Exception("Cannot load the dataloader for further process".capitalize())


def helpers(**kwargs):
    in_channels = kwargs["in_channels"]
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    device = kwargs["device"]

    netG_XtoY = Generator(in_channels=in_channels)
    netG_YtoX = Generator(in_channels=in_channels)

    netD_X = Discriminator(in_channels=in_channels)
    netD_Y = Discriminator(in_channels=in_channels)

    if adam:
        optimizerG = optim.Adam(
            params=list(netG_XtoY.parameters()) + list(netG_YtoX.parameters()),
            lr=lr,
            betas=(0.5, 0.999),
        )
        optimizerD_X = optim.Adam(
            params=netD_X.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
        )
        optimizerD_Y = optim.Adam(
            params=netD_Y.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
        )

    elif SGD:
        optimizerG = optim.SGD(
            params=list(netG_XtoY.parameters()) + list(netG_YtoX.parameters()),
            lr=lr,
            momentum=0.95,
        )
        optimizerD_X = optim.SGD(params=netD_X.parameters(), lr=lr, momentum=0.95)
        optimizerD_Y = optim.SGD(params=netD_Y.parameters(), lr=lr, momentum=0.95)

    try:
        dataloader = load_dataloader()

    except Exception as e:
        print("An error occurred {}".format(e))

    cycle_loss = CycleLoss(reduction="mean")
    grad_penalty = GradientPenalty(in_channels=in_channels, batch_size=1, device=device)

    return {
        "optimizerG": optimizerG,
        "optimizerD_X": optimizerD_X,
        "optimizerD_Y": optimizerD_Y,
        "netG_XtoY": netG_XtoY,
        "netG_YtoX": netG_YtoX,
        "netD_X": netD_X,
        "netD_Y": netD_Y,
        "dataloader": dataloader,
        "cycle_loss": cycle_loss,
        "grad_penalty": grad_penalty,
        "train_dataloader": dataloader["train_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
    }


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    init = helpers(
        lr=0.001,
        adam=True,
        SGD=False,
        in_channels=3,
        device=device,
    )

    gp = init["grad_penalty"]
    netD = Discriminator(in_channels=3).to(device)

    X = torch.randn(1, 3, 256, 256).to(device)
    y = torch.randn(1, 3, 256, 256).to(device)

    print(gp(netD, X, y, device=device))
