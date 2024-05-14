import sys
import os
import torch
import unittest

sys.path.append("src/")

from dataloader import Loader
from utils import load, config
from discriminator import Discriminator
from generator import Generator
from helper import helpers


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.train_dataloader = load(
            filename=os.path.join(
                config()["path"]["processed_path"], "train_dataloader.pkl"
            )
        )
        self.test_dataloader = load(
            filename=os.path.join(
                config()["path"]["processed_path"], "test_dataloader.pkl"
            )
        )

        self.netD = Discriminator(in_channels=3)
        self.netG = Generator(in_channels=3)
        self.init = helpers(
            lr=0.0002,
            adam=True,
            SGD=False,
            in_channels=3,
            device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        )

    def test_total_train_data(self):
        self.assertEqual(sum(X.size(0) for X, _ in self.train_dataloader), 18)

    def test_total_test_data(self):
        self.assertEqual(sum(X.size(0) for X, _ in self.test_dataloader), 7)

    def test_train_data_shape(self):
        self.assertEqual(self.train_dataloader.dataset[0][0].shape, (3, 256, 256))

    def test_test_data_shape(self):
        self.assertEqual(self.test_dataloader.dataset[0][0].shape, (3, 256, 256))

    def test_total_data(self):
        total_data = sum(X.size(0) for X, _ in self.train_dataloader) + sum(
            X.size(0) for X, _ in self.test_dataloader
        )
        self.assertEqual(total_data, 25)

    def test_netD_shape(self):
        self.assertEqual(self.netD(torch.randn(1, 3, 256, 256)).size(), (1, 1, 30, 30))

    def test_netG_shape(self):
        self.assertEqual(
            self.netG(torch.randn(1, 3, 256, 256)).size(), (1, 3, 256, 256)
        )

    def test_init_train_dataloader_size(self):
        self.assertEqual(sum(X.size(0) for X, _ in self.init["train_dataloader"]), 18)

    def test_init_train_dataloader_size(self):
        self.assertEqual(sum(X.size(0) for X, _ in self.init["test_dataloader"]), 7)

    def test_init_cycle_loss(self):
        actual = torch.tensor([1.0, 0.0, 1.0])
        predicted = torch.tensor([1.0, 0.0, 1.0])

        self.assertEqual(self.init["cycle_loss"](actual, predicted), 0.0)

    def test_gradient_penalty(self):
        self.gp = self.init["grad_penalty"]
        result = self.gp(
            self.netD,
            torch.randn(1, 3, 256, 256),
            torch.randn(1, 3, 256, 256),
            device="cpu",
        )
        self.assertGreater(result, 0.001)

    def test_total_quantity(self):
        train_dataloader = self.init["train_dataloader"]
        test_dataloader = self.init["test_dataloader"]

        total_quantity = sum(X.size(0) for X, _ in train_dataloader) + sum(
            X.size(1) for X, _ in test_dataloader
        )

        self.assertEqual(total_quantity, 21)


if __name__ == "__main__":
    unittest.main()
