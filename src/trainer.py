import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from helper import helpers
from generator import Generator
from utils import config, device_init, weight_init, dump, load


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

        self.device = device_init(device=self.device)

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
            print("An Error occurred in the code:", e)
        else:
            self.netG_XtoY_path = self.config_files["path"]["netG_XtoY_path"]
            self.netG_YtoX_path = self.config_files["path"]["netG_YtoX_path"]
            self.best_model_path = self.config_files["path"]["best_model_path"]
            self.train_results = self.config_files["path"]["train_results"]
            self.metrics_path = self.config_files["path"]["metrics_path"]

        self.loss = float("inf")

        self.total_netG_loss = []
        self.total_netD_X_loss = []
        self.total_netD_Y_loss = []

        self.history = {"netG_loss": [], "netD_X_loss": [], "netD_Y_loss": []}

    def l1(self, model):
        if isinstance(model, Generator):
            return sum(torch.norm(params, 1) for params in model.parameters())

        else:
            raise ValueError("Model is not a Generator".capitalize())

    def l2(self, model):
        if isinstance(model, Generator):
            return sum(torch.norm(params, 2) for params in model.parameters())

        else:
            raise ValueError("Model is not a Generator".capitalize())

    def elastic_net(self, model):
        if isinstance(model, Generator):
            return self.l1(model) + self.l2(model)

        else:
            raise ValueError("Model is not a Generator".capitalize())

    def update_netG(self, **kwargs):
        self.optimizerG.zero_grad()

        fake_y = self.netGX_toY(kwargs["X"])
        fake_y_loss = -torch.mean(self.netD_Y(fake_y))
        reconstructed_X = self.netGY_toX(fake_y)

        fake_x = self.netGY_toX(kwargs["y"])
        fake_x_loss = -torch.mean(self.netD_X(fake_x))
        reconstructed_y = self.netGX_toY(fake_x)

        total_W_loss = 1.0 * (fake_x_loss + fake_y_loss)

        cycle_loss_X = self.cycle_loss(kwargs["X"], reconstructed_X).mean()
        cycle_loss_y = self.cycle_loss(kwargs["y"], reconstructed_y).mean()

        total_cycle_loss = 10.0 * (cycle_loss_X + cycle_loss_y)

        pixel_loss_X = torch.abs(fake_x - kwargs["X"]).mean()
        pixel_loss_y = torch.abs(fake_y - kwargs["y"]).mean()

        total_content_loss = 10.0 * (pixel_loss_X + pixel_loss_y)

        total_G_loss = total_W_loss + total_cycle_loss + total_content_loss

        total_G_loss.backward()
        self.optimizerG.step()

        return total_G_loss.item()

    def update_netD_X(self, **kwargs):
        self.optimizerD_X.zero_grad()

        fake_x = self.netGY_toX(kwargs["y"]).detach()
        real_x_predict = self.netD_X(kwargs["X"])
        fake_x_predict = self.netD_X(fake_x)
        grad_X_loss = self.grad_penalty(
            self.netD_X, kwargs["X"], kwargs["y"], device=self.device
        )
        netD_X_loss = (
            -torch.mean(real_x_predict)
            + torch.mean(fake_x_predict)
            + (5.0 * grad_X_loss)
        )

        netD_X_loss.backward()
        self.optimizerD_X.step()

        return netD_X_loss.item()

    def update_netD_Y(self, **kwargs):
        self.optimizerD_Y.zero_grad()

        fake_y = self.netGX_toY(kwargs["X"]).detach()
        real_y_predict = self.netD_Y(kwargs["y"])
        fake_y_predict = self.netD_Y(fake_y)
        grad_Y_loss = self.grad_penalty(
            self.netD_Y, kwargs["X"], kwargs["y"], device=self.device
        )
        netD_Y_loss = (
            -torch.mean(real_y_predict)
            + torch.mean(fake_y_predict)
            + (5.0 * grad_Y_loss)
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
                "Epochs - [{}/{}] is completed".format(
                    kwargs["epoch"],
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
            raise Exception("Cannot save the model")

    def early_stopping(self):
        pass

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
                normalize=True,
                nrow=4,
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

            if self.lr_scheduler:
                self.schedulerG.step()
                self.schedulerD_X.step()
                self.schedulerD_Y.step()

            self.show_progress(
                epoch=epoch + 1,
                netG_loss=netG_loss,
                netD_X_loss=netD_X_loss,
                netD_Y_loss=netD_Y_loss,
            )

            self.saved_checkpoints(epoch=epoch + 1, netG_loss=np.mean(netG_loss))

            if (epoch + 1) % 50:
                self.saved_training_results(epoch=epoch + 1)

            self.total_netG_loss.append(np.mean(netG_loss))
            self.total_netD_X_loss.append(np.mean(netD_X_loss))
            self.total_netD_Y_loss.append(np.mean(netD_Y_loss))

        self.history["netG_loss"].append(self.total_netG_loss)
        self.history["netD_X_loss"].append(self.total_netD_X_loss)
        self.history["netD_Y_loss"].append(self.total_netD_Y_loss)

        for filename, file in self.history.items():

            dump(
                value=file,
                filename=os.path.join(self.metrics_path, "{}.pkl".format(filename)),
            )

    @staticmethod
    def plot_history():
        config_files = config()
        metrics_path = config_files["path"]["metrics_path"]

        plt.figure(figsize=(20, 10))

        if os.path.exists(metrics_path):
            history = load(os.path.join(metrics_path, "history.pkl"))
            for index, (filename, loss) in enumerate(history.items()):
                plt.subplot(1, 3, index + 1)

                plt.plot(loss)
                plt.title(filename)

                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(metrics_path, "history.png"))
            plt.show()

        else:
            raise Exception("Cannot save the metrics".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for DualGAN")
    parser.add_argument("--in_channels", type=int, default=3, help="Input channels")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Initial learning rate"
    )
    parser.add_argument(
        "--num_critics",
        type=int,
        default=5,
        help="Number of critic iterations per generator iteration",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Size of the batches"
    )
    parser.add_argument("--adam", type=bool, default=True, help="Use Adam optimizer")
    parser.add_argument("--SGD", type=bool, default=False, help="Use SGD optimizer")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--is_weight_init", type=bool, default=True, help="Use weight initialization"
    )
    parser.add_argument(
        "--lr_scheduler", type=bool, default=False, help="Use learning rate scheduler"
    )
    parser.add_argument(
        "--is_display", type=bool, default=True, help="Display progress during training"
    )
    args = parser.parse_args()

    trainer = Trainer(
        in_channels=args.in_channels,
        epochs=args.epochs,
        lr=args.lr,
        num_critics=args.num_critics,
        device=args.device,
        adam=args.adam,
        SGD=args.SGD,
        is_weight_init=args.is_weight_init,
        is_display=args.is_display,
        lr_scheduler=args.lr_scheduler,
    )

    trainer.train()
    trainer.plot_history()
