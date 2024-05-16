import sys
import os
import torch
import imageio
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


sys.path.append("src/")

from utils import config, device_init, load
from generator import Generator


class TestModel:
    def __init__(
        self,
        in_channels=3,
        dataloader="test",
        best_model=True,
        XtoY=None,
        YtoX=None,
        device="cuda",
        create_gif=False,
    ):
        self.in_channels = in_channels
        self.dataloader = dataloader
        self.best_model = best_model
        self.XtoY = XtoY
        self.YtoX = YtoX
        self.device = device_init(device=device)
        self.create_gif_images = create_gif

        self.netG_XtoY = Generator(in_channels=3).to(self.device)
        self.netG_YtoX = Generator(in_channels=3).to(self.device)

        self.config_files = config()

    def select_best_model(self):
        if self.best_model:
            best_model_path = self.config_files["path"]["best_model_path"]
            if os.path.exists(best_model_path):
                state_dict = torch.load(os.path.join(best_model_path, "best_model.pth"))

                self.netG_XtoY.load_state_dict(state_dict["netG_XtoY"])
                self.netG_YtoX.load_state_dict(state_dict["netG_YtoX"])

        else:
            if isinstance(self.XtoY, str) and isinstance(self.YtoX, str):
                state_XtoY = torch.load(self.XtoY)
                state_YtoX = torch.load(self.YtoX)

                self.netG_XtoY.load_state_dict(state_XtoY["netG_XtoY"])
                self.netG_YtoX.load_state_dict(state_YtoX["netG_YtoX"])

    def load_dataloader(self):
        if self.dataloader == "test":
            dataloader = load(
                filename=os.path.join(
                    self.config_files["path"]["processed_path"], "test_dataloader.pkl"
                )
            )
        elif self.dataloader == "train":
            dataloader = load(
                filename=os.path.join(
                    self.config_files["path"]["processed_path"], "train_dataloader.pkl"
                )
            )
        else:
            raise ValueError("Invalid dataloader")

        return dataloader

    def normalize_image(self, image):
        return (image - image.min()) / (image.max() - image.min())

    def plot(self, dataloader=None):
        if isinstance(dataloader, DataLoader):
            dataloader = dataloader

            plt.figure(figsize=(10, 12))

            X, y = next(iter(dataloader))

            X = X.to(self.device)
            y = y.to(self.device)

            predict_Y = self.netG_XtoY(X)
            reconstructed_X = self.netG_YtoX(predict_Y)

            for index, image in enumerate(predict_Y):
                pred_y = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                pred_y = self.normalize_image(pred_y)

                revert_X = (
                    reconstructed_X[index]
                    .squeeze()
                    .permute(1, 2, 0)
                    .cpu()
                    .detach()
                    .numpy()
                )
                revert_X = self.normalize_image(revert_X)

                real_X = X[index].permute(1, 2, 0).cpu().detach().numpy()
                real_X = self.normalize_image(real_X)

                real_Y = y[index].permute(1, 2, 0).cpu().detach().numpy()
                real_Y = self.normalize_image(real_Y)

                for idx, (title, image) in enumerate(
                    [
                        ("real X", real_X),
                        ("pred Y", pred_y),
                        ("real Y", real_Y),
                        ("revert X", revert_X),
                    ]
                ):

                    plt.subplot(4 * 2, 4 * 2, 4 * index + (idx + 1))
                    plt.imshow(image)
                    plt.title(title)
                    plt.axis("off")

            plt.savefig(
                os.path.join(
                    self.config_files["path"]["test_result"], "test_result.jpeg"
                )
            )
            print(
                "Test images saved in the folder: ",
                self.config_files["path"]["test_result"],
            )
            plt.show()
            plt.close()

        else:
            raise ValueError("Invalid dataloader".capitalize())

    def create_gif(self):
        if self.create_gif_images:
            train_images = self.config_files["path"]["train_results"]
            gif_path = self.config_files["path"]["gif_path"]
            images = [
                imageio.imread(os.path.join(train_images, image))
                for image in os.listdir(train_images)
            ]

            imageio.mimsave(os.path.join(gif_path, "gif.gif"), images, "GIF")

        else:
            pass

    def test(self):
        try:
            self.select_best_model()
        except Exception as e:
            print(f"An error occurred while selecting the best model: {e}")
            return

        try:
            dataloader = self.load_dataloader()
        except Exception as e:
            print(f"An error occurred while loading the dataloader: {e}")
            return

        try:
            self.plot(dataloader=dataloader)
        except Exception as e:
            print(f"An error occurred while plotting: {e}")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the modelfor DualGAN".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Define the image channels".capitalize(),
    )
    parser.add_argument(
        "--best_model",
        type=bool,
        default=True,
        help="Select the best model".capitalize(),
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="test",
        help="Select the dataloader".capitalize(),
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Select the device".capitalize()
    )
    parser.add_argument(
        "--XtoY", type=str, help="Denine the netG_XtoY model".capitalize()
    )
    parser.add_argument(
        "--YtoX", type=str, help="Denine the netG_YtoX model".capitalize()
    )
    parser.add_argument(
        "--create_gif", type=bool, default=True, help="Create a gif".capitalize()
    )

    args = parser.parse_args()

    test_model = TestModel(
        in_channels=args.in_channels,
        dataloader=args.dataloader,
        device=args.device,
        best_model=args.best_model,
        create_gif=args.create_gif,
        XtoY=args.XtoY,
        YtoX=args.YtoX,
    )
    test_model.test()
    test_model.create_gif()
