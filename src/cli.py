import sys
import yaml
import argparse

sys.path.append("src/")

from dataloader import Loader
from trainer import Trainer
from test import TestModel


def cli():
    parser = argparse.ArgumentParser(description="CLI for DualGAN".capitalize())
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
    parser.add_argument("--adam", type=bool, default=True, help="Use Adam optimizer")
    parser.add_argument("--SGD", type=bool, default=False, help="Use SGD optimizer")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
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
        "--XtoY", type=str, help="Denine the netG_XtoY model".capitalize()
    )
    parser.add_argument(
        "--YtoX", type=str, help="Denine the netG_YtoX model".capitalize()
    )
    parser.add_argument(
        "--create_gif", type=bool, default=False, help="Create a gif".capitalize()
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )
    args = parser.parse_args()

    if args.train:

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

        trainer = Trainer(
            in_channels=args.channels,
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

        with open("./trained_params.yml", "w") as file:
            yaml.safe_dump(
                {
                    "in_channels": args.channels,
                    "image_size": args.image_size,
                    "channels": args.channels,
                    "split_size": args.split_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "num_critics": args.num_critics,
                    "batch_size": args.batch_size,
                    "adam": args.adam,
                    "SGD": args.SGD,
                    "is_weight_init": args.is_weight_init,
                    "lr_scheduler": args.lr_scheduler,
                    "is_display": args.is_display,
                },
                file,
            )

    elif args.test:
        test_model = TestModel(
            in_channels=args.channels,
            dataloader=args.dataloader,
            device=args.device,
            best_model=args.best_model,
            create_gif=args.create_gif,
            XtoY=args.XtoY,
            YtoX=args.YtoX,
        )
        test_model.test()
        test_model.create_gif()


if __name__ == "__main__":
    cli()
