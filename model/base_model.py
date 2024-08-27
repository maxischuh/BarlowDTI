from typing import Tuple, Any, Union
import torch
from torch import nn
import numpy as np


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # set device (gpu 0 or 1 if available or cpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # make empty param dict
        self.param_dict = {}

        # make optimizer options dict
        self.optimizer_dict = {
            "adam": torch.optim.Adam,
            "nadam": torch.optim.NAdam,
            "adamax": torch.optim.Adamax,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }

        # make loss options dict
        self.loss_dict = {
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "smoothl1": nn.SmoothL1Loss,
            "huber": nn.HuberLoss,
            "cel": nn.CrossEntropyLoss,  # Suitable for classification tasks
            "bcel": nn.BCELoss,  # Suitable for classification tasks
        }

        # make activation function options dictionary
        self.activation_dict = {
            "relu": nn.ReLU,
            "swish": nn.Hardswish,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
        }

        # make tokenizer placeholder
        self.tokenizer = None

        # create history dictionary
        self.history = {
            "train_loss": [],
            "on_diag_loss": [],
            "off_diag_loss": [],
            "validation_loss": [],
            "learning_rate": [],
        }

        # create early stopping params
        self.count = 0

    def print_config(self) -> None:
        print("[BT]: Current parameter config:")
        print(self.param_dict)

    def early_stopping(self, patience: int) -> bool:
        # count every epoch that's worse than the best for patience times
        if len(self.history["validation_loss"]) > patience:
            best_loss = min(self.history["validation_loss"])
            if self.history["validation_loss"][-1] > best_loss:
                self.count += 1
            else:
                self.count = 0
            if self.count >= patience:
                if self.param_dict["verbose"] is True:
                    print("[BT]: Early stopping")
                return True
        return False
