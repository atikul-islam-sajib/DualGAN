import sys
import os

sys.path.append("src/")

from helper import helpers
from utils import config, device_init, load, dump


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
            pass
