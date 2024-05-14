import sys
import os

from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from helper import helpers
from utils import config, device_init, weight_init, load, dump


class Trainer:
    def __init__(
        self,
        in_channels=3,
        epochs=500,
        lr=2e-3,
        device="cuda",
        adam=True,
        SGD=False,
        is_weight_init=True,
        lr_scheduler=False,
    ):
        self.in_channels = in_channels
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.is_weight_init = is_weight_init
        self.lr_scheduler = lr_scheduler

        self.init = helpers(
            in_channels=in_channels,
            lr=lr,
            adam=adam,
            SGD=SGD,
            device=device_init(device=device),
        )

        self.netGX_toY = self.init["netG_XtoY"]
        self.netGY_toX = self.init["netG_YtoX"]

        self.netDX = self.init["netD_X"]
        self.netDY = self.init["netD_Y"]

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD_X = self.init["optimizerD_X"]
        self.optimizerD_Y = self.init["optimizerD_Y"]

        self.train_dataloader = self.init["train_dataloader"]
        self.test_dataloader = self.init["test_dataloader"]

        self.cycle_loss = self.init["cycle_loss"]
        self.grad_penalty = self.init["grad_penalty"]

        if self.is_weight_init:
            self.netGX_toY.apply(weight_init)
            self.netGY_toX.apply(weight_init)

            self.netDX.apply(weight_init)
            self.netDY.apply(weight_init)

        if self.lr_scheduler:
            self.schedulerG = StepLR(
                optimizer=self.optimizerG,
                step_size=20,
                gamma=0.5,
            )
            self.schedulerD_X = StepLR(
                optimizer=self.optimizerD_X,
                step_size=20,
                gamma=0.5,
            )
            self.schedulerD_Y = StepLR(
                optimizer=self.optimizerD_Y,
                step_size=20,
                gamma=0.5,
            )

        print(self.netGX_toY)
        print(self.netGY_toX)
        print(self.netDX)
        print(self.netDY)
        print(self.optimizerG)
        print(self.optimizerD_X)
        print(self.optimizerD_Y)
        print(self.train_dataloader)
        print(self.test_dataloader)
        print(self.cycle_loss)
        print(self.grad_penalty)


if __name__ == "__main__":
    trainer = Trainer(
        in_channels=3,
        epochs=1,
        lr=0.0002,
        device="mps",
        adam=True,
        SGD=False,
        is_weight_init=True,
        lr_scheduler=False,
    )
