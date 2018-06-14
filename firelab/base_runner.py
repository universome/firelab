import os
from os import path
import random
from itertools import islice

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np; np.random.seed(42)
from tqdm import tqdm


class BaseRunner:
    def __init__(self, config):
        self.config = config

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def evaluate(self, dataloader):
        raise NotImplementedError

    def train(self):
        pass

    def start(self):
        self.init_dataloaders()
        self.trainer.run_training()
        # self.evaluate(self.test_dataloader())

    def pause_training(self):
        """Pauses the training process"""
        # TODO: pause/continue evaluation process
        raise NotImplementedError

    def continue_training(self):
        """Continues training after the pause"""
        raise NotImplementedError
