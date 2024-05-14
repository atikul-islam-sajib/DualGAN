import os
import yaml
import torch
import torch.nn as nn
import joblib


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise Exception("Please provide the value and filename to dump".capitalize())


def load(filename):
    if filename is not None:
        return joblib.load(filename=filename)


def config():
    with open("./config.yml", "r") as file:
        config_files = yaml.safe_load(file)

    return config_files


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        return torch.device("cuda")


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
