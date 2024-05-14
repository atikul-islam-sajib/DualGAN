import sys
import os
import torch
import unittest

sys.path.append("src/")

from dataloader import Loader
from utils import load, config
from discriminator import Discriminator


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


if __name__ == "__main__":
    unittest.main()
