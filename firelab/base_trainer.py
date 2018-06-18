import os

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from firelab.utils import cudable

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs')
        self.max_num_iters = config.get('max_num_iters')
        self.val_freq = config.get('val_freq')
        self.early_stopping_last_n_iters = config.get('early_stopping_last_n_iters')

        self.val_freq = config.get('val_freq')
        self.checkpoint_freq = config.get('checkpoint_freq', 100)

        self.train_dataloader = None
        self.val_dataloader = None

        self.writer = SummaryWriter(config['firelab']['logs_path'])

    def start(self):
        self.init_dataloaders()
        self.init_models()
        self.run_training()

    def init_dataloaders(self):
        pass

    def init_models(self):
        pass

    def run_training(self):
        while not self.should_stop():
            try:
                for batch in tqdm(self.train_dataloader, leave=False):
                    batch = cudable(batch)
                    self.train_on_batch(batch)
                    self.num_iters_done += 1
                    self.log_scores()
                    self.try_to_validate()
                    self.try_to_checkpoint()

                self.num_epochs_done += 1
            except KeyboardInterrupt:
                print('\nTerminating experiment...')
                break

    def train_on_batch(self, batch):
        pass

    def log_scores(self):
        pass

    def try_to_validate(self):
        if self.val_freq and self.num_iters_done % self.val_freq == 0:
            self.validate()

    def validate(self):
        pass

    def try_to_checkpoint(self):
        if self.checkpoint_freq and self.num_iters_done % self.checkpoint_freq == 0:
            self.checkpoint()

    def checkpoint(self):
        pass

    def should_stop(self):
        if self.max_num_iters and self.num_iters_done >= self.max_num_iters: return True
        if self.max_num_epochs and self.num_epochs_done >= self.max_num_epochs: return True
        if self.should_early_stop(): return True

        return False

    def should_early_stop(self):
        """Checks early stopping criteria"""
        return False

    def train_mode(self):
        pass

    def test_mode(self):
        pass

    def save_model(self, model, name):
        model_name = '{}-{}.pth'.format(name, self.num_iters_done)
        model_path = os.path.join(self.config['firelab']['checkpoints_path'], model_name)
        torch.save(model.state_dict(), model_path)
