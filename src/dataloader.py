import os
import sys
import cv2
import zipfile
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("src/")

from utils import dump, load, config


class Loader:
    def __init__(
        self,
        image_path=None,
        channels=3,
        image_size=256,
        batch_size=1,
        split_size=0.20,
        paired_images=False,
        unpaired_images=True,
    ):
        self.image_path = image_path
        self.channels = channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size
        self.paired_images = paired_images
        self.unpaired_images = unpaired_images

        self.X = []
        self.y = []

        self.raw_path = config()["path"]["raw_path"]
        self.processed_path = config()["path"]["processed_path"]

    def unzip_folder(self):
        with zipfile.ZipFile(self.image_path, "r") as file:
            if os.path.exists(self.raw_path):
                file.extractall(os.path.join(self.raw_path))

            else:
                raise Exception(
                    "Cannot unzip the folder for further process".capitalize()
                )

    def transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def image_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.split_size, random_state=42
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def feature_extractor(self):
        self.directory = os.path.join(self.raw_path, "images")
        self.categories = ["X", "y"]

        self.paired_check = os.listdir(os.path.join(self.directory, "y"))

        for category in tqdm(self.categories):
            folder_path = os.path.join(self.directory, category)

            for image in os.listdir(folder_path):
                if self.paired_images:
                    if image in self.paired_check:
                        image_path = os.path.join(folder_path, image)
                else:
                    image_path = os.path.join(folder_path, image)

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = self.transforms()(Image.fromarray(image))

                self.X.append(image) if category == "X" else self.y.append(image)

        return self.image_split(X=self.X, y=self.y)

    def create_dataloader(self):
        try:
            data = self.feature_extractor()

        except Exception as e:
            print("An error occurred while creating the dataloader".capitalize(), e)

        else:
            train_dataloader = DataLoader(
                dataset=list(zip(data["X_train"], data["y_train"])),
                batch_size=self.batch_size,
                shuffle=True,
            )
            test_dataloader = DataLoader(
                dataset=list(zip(data["X_test"], data["y_test"])),
                batch_size=self.batch_size * 8,
                shuffle=True,
            )

            if os.path.exists(self.processed_path):
                dump(
                    value=train_dataloader,
                    filename=os.path.join(self.processed_path, "train_dataloader.pkl"),
                )
                dump(
                    value=test_dataloader,
                    filename=os.path.join(self.processed_path, "test_dataloader.pkl"),
                )

            else:
                raise Exception(
                    "Cannot create the dataloader for further process".capitalize()
                )

    @staticmethod
    def plot_images():
        config_files = config()
        files_path = config_files["path"]["files_path"]
        processed_path = config_files["path"]["processed_path"]

        if os.path.exists(files_path):
            test_dataloader = load(
                filename=os.path.join(processed_path, "test_dataloader.pkl")
            )

            X, y = next(iter(test_dataloader))

        else:
            raise Exception(
                "Cannot load the dataloader for further process".capitalize()
            )

        plt.figure(figsize=(20, 10))

        for index, image in enumerate(X):
            image_X = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            image_y = y[index].squeeze().permute(1, 2, 0).cpu().detach().numpy()

            image_X = (image_X - image_X.min()) / (image_X.max() - image_X.min())
            image_y = (image_y - image_y.min()) / (image_y.max() - image_y.min())

            plt.subplot(2 * 2, 2 * 4, 2 * index + 1)
            plt.imshow(image_X)
            plt.axis("off")
            plt.title("X")

            plt.subplot(2 * 2, 2 * 4, 2 * index + 2)
            plt.imshow(image_y)
            plt.axis("off")
            plt.title("y")

        plt.tight_layout()
        (
            plt.savefig(os.path.join(files_path, "images.png"))
            if os.path.exists(files_path)
            else "Cannot be saved the images".capitalize()
        )
        plt.show()

    @staticmethod
    def dataset_details():
        config_files = config()

        files_path = config_files["path"]["files_path"]

        train_dataloader = load(
            os.path.join(config_files["path"]["processed_path"], "train_dataloader.pkl")
        )
        test_dataloader = load(
            os.path.join(config_files["path"]["processed_path"], "test_dataloader.pkl")
        )

        pd.DataFrame(
            {
                "train_data(total)": str(sum(X.size(0) for X, _ in train_dataloader)),
                "test_data(total)": str(sum(X.size(0) for X, _ in test_dataloader)),
                "total_data": str(
                    sum(X.size(0) for X, _ in train_dataloader)
                    + sum(X.size(0) for X, _ in test_dataloader)
                ),
                "train_data_shape": str(train_dataloader.dataset[0][0].shape),
                "test_data_shape": str(test_dataloader.dataset[0][0].shape),
            },
            index=["Quantity"],
        ).T.to_csv(
            os.path.join(files_path, "dataset_details.csv")
            if os.path.exists(files_path)
            else "Cannot be saved the dataset into csv format".capitalize()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader for DualGAN".capitalize())
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Define the channels to load".capitalize(),
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Define the image size".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Define the batch size".capitalize()
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=0.20,
        help="Define the split size".capitalize(),
    )
    parser.add_argument(
        "--paired_images",
        type=bool,
        default=False,
        help="Define the paired images".capitalize(),
    )
    parser.add_argument(
        "--unpaired_images",
        type=bool,
        default=True,
        help="Define the unpaired images".capitalize(),
    )
    args = parser.parse_args()

    loader = Loader(
        image_path=args.image_path,
        channels=args.channels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        split_size=args.split_size,
        paired_images=args.paired_images,
        unpaired_images=args.unpaired_images,
    )
    # loader.unzip_folder()
    loader.create_dataloader()
    loader.plot_images()
    loader.dataset_details()
