import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
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
        num_critics=5,
        device="cuda",
        adam=True,
        SGD=False,
        is_weight_init=True,
        is_display=True,
        lr_scheduler=False,
    ):
        self.in_channels = in_channels
        self.epochs = epochs
        self.lr = lr
        self.num_critics = num_critics
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.is_weight_init = is_weight_init
        self.is_display = is_display
        self.lr_scheduler = lr_scheduler

        self.device = device_init(
            device=self.device,
        )

        self.init = helpers(
            in_channels=in_channels,
            lr=lr,
            adam=adam,
            SGD=SGD,
            device=device_init(device=device),
        )

        self.netGX_toY = self.init["netG_XtoY"].to(self.device)
        self.netGY_toX = self.init["netG_YtoX"].to(self.device)

        self.netD_X = self.init["netD_X"].to(self.device)
        self.netD_Y = self.init["netD_Y"].to(self.device)

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

            self.netD_X.apply(weight_init)
            self.netD_Y.apply(weight_init)

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
        try:
            self.config_files = config()
        except Exception as e:
            print("An Error occur in the code:".capitalize(), e)

        else:
            self.netG_XtoY_path = self.config_files["path"]["netG_XtoY_path"]
            self.netG_YtoX_path = self.config_files["path"]["netG_YtoX_path"]
            self.best_model_path = self.config_files["path"]["best_model_path"]
            self.train_results = self.config_files["path"]["train_results"]

        self.loss = float("inf")

        self.total_netG_loss = []
        self.total_netD_X_loss = []
        self.total_netD_Y_loss = []

    def update_netG(self, **kwargs):
        self.optimizerG.zero_grad()

        fake_y = self.netGX_toY(kwargs["X"])
        fake_y_loss = -torch.mean(fake_y)
        reconstructed_X = self.netGY_toX(fake_y)

        fake_x = self.netGY_toX(kwargs["y"])
        fake_x_loss = -torch.mean(fake_x)
        reconstructed_y = self.netGX_toY(fake_x)

        total_W_loss = 1.0 * (fake_x_loss + fake_y_loss)

        cycle_loss_X = self.cycle_loss(kwargs["X"], reconstructed_X)
        cycle_loss_y = self.cycle_loss(kwargs["y"], reconstructed_y)

        total_cycle_loss = 10.0 * (cycle_loss_X + cycle_loss_y)

        total_G_loss = total_W_loss + total_cycle_loss

        total_G_loss.backward()
        self.optimizerG.step()

        return total_G_loss.item()

    def update_netD_X(self, **kwargs):
        self.optimizerD_X.zero_grad()

        fake_x = self.netGY_toX(kwargs["y"])
        real_x_predict = self.netD_X(kwargs["X"])
        grad_X_loss = self.grad_penalty(
            self.netD_X, kwargs["y"], kwargs["X"], device=self.device
        )
        netD_X_loss = (
            -torch.mean(real_x_predict) + torch.mean(fake_x) + 10.0 * grad_X_loss
        )

        netD_X_loss.backward()
        self.optimizerD_X.step()

        return netD_X_loss.item()

    def update_netD_Y(self, **kwargs):
        self.optimizerD_Y.zero_grad()

        fake_y = self.netGX_toY(kwargs["X"])
        real_y_predict = self.netD_Y(kwargs["y"])
        grad_Y_loss = self.grad_penalty(
            self.netD_Y, kwargs["X"], kwargs["y"], device=self.device
        )

        netD_Y_loss = (
            -torch.mean(real_y_predict) + torch.mean(fake_y) + 10.0 * grad_Y_loss
        )

        netD_Y_loss.backward()
        self.optimizerD_Y.step()

        return netD_Y_loss.item()

    def show_progress(self, **kwargs):
        if self.is_display:
            print(
                "Epochs - [{}/{}] - netG_loss: {} - netD_X_loss: {} - netD_Y_loss: {}".format(
                    kwargs["epoch"],
                    self.epochs,
                    np.mean(kwargs["netG_loss"]),
                    np.mean(kwargs["netD_X_loss"]),
                    np.mean(kwargs["netD_Y_loss"]),
                )
            )
        else:
            print(
                "Epochs - [{}/{}] is completed".capitalize().format(
                    kwargs["epochs"],
                    self.epochs,
                )
            )

    def saved_checkpoints(self, **kwargs):
        if (
            os.path.exists(self.netG_XtoY_path)
            and os.path.exists(self.netG_YtoX_path)
            and os.path.exists(self.best_model_path)
        ):
            for filename, model, path in [
                ("netG_XtoY", self.netGX_toY, self.netG_XtoY_path),
                ("netG_YtoX", self.netGY_toX, self.netG_YtoX_path),
            ]:
                torch.save(
                    model.state_dict(),
                    os.path.join(path, "{}{}.pth".format(filename, kwargs["epoch"])),
                )

            if self.loss > kwargs["netG_loss"]:
                self.loss = kwargs["netG_loss"]

                torch.save(
                    {
                        "netG_XtoY": self.netGX_toY.state_dict(),
                        "netG_YtoX": self.netGY_toX.state_dict(),
                        "netG_loss": kwargs["netG_loss"],
                        "epoch": kwargs["epoch"],
                    },
                    os.path.join(self.best_model_path, "best_model.pth"),
                )

        else:
            raise Exception("Cannot be saved the model")

    def saved_training_results(self, **kwargs):
        X, y = next(iter(self.test_dataloader))

        predict_y = self.netGX_toY(X.to(self.device))
        reconstructed_X = self.netGY_toX(predict_y.to(self.device))

        for filename, samples in [
            ("predict_y", predict_y),
            ("reconstructed_X", reconstructed_X),
        ]:
            save_image(
                samples,
                os.path.join(
                    self.train_results, "{}{}.png".format(filename, kwargs["epoch"])
                ),
            )

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            netG_loss = []
            netD_X_loss = []
            netD_Y_loss = []

            for idx, (X, y) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                netD_X_loss.append(self.update_netD_X(X=X, y=y))
                netD_Y_loss.append(self.update_netD_Y(X=X, y=y))

                if (idx + 1) % self.num_critics:
                    netG_loss.append(self.update_netG(X=X, y=y))

            self.show_progress(
                epoch=epoch + 1,
                netG_loss=netG_loss,
                netD_X_loss=netD_X_loss,
                netD_Y_loss=netD_Y_loss,
            )

            self.saved_checkpoints(epoch=epoch + 1, netG_loss=np.mean(netG_loss))
            self.saved_training_results(epoch=epoch + 1)


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
    trainer.train()
