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

        if 'checkpoint' in self.config:
            self.checkpoint_freq = self.config['checkpoint'].get('freq_iters')
            self.checkpoint_freq_epochs = self.config['checkpoint'].get('freq_epochs')
            self.checkpoint_list = self.config['checkpoint']['modules']

        assert not (self.checkpoint_freq and self.checkpoint_freq_epochs), """
            Can't save both on iters and epochs"""

        self.train_dataloader = None
        self.val_dataloader = None

        self.writer = SummaryWriter(config['firelab']['logs_path'])

    def start(self):
        self.init_dataloaders()
        self.init_models()
        self.init_criterions()
        self.init_optimizers()
        self.load_checkpoint()

        self.run_training()

    def init_dataloaders(self):
        pass

    def init_models(self):
        pass

    def init_criterions(self):
        pass

    def init_optimizers(self):
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
                    self.checkpoint()

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

    def checkpoint(self):
        should_checkpoint = False

        if self.checkpoint_freq:
            should_checkpoint = self.num_iters_done % self.checkpoint_freq == 0
        elif self.checkpoint_freq_epochs:
            # TODO: looks like govnokod
            epoch_size = len(self.train_dataloader)
            freq = self.checkpoint_freq_epochs * epoch_size
            should_checkpoint = self.num_iters_done % freq == 0

        if not should_checkpoint: return

        for module_name in self.checkpoint_list:
            self.save_module_state(getattr(self, module_name), module_name)

        self.checkpoint_freq_warning()

    def checkpoint_freq_warning(self):
        """
        Prints warning if we write checkpoints too often
        TODO: wip
        """
        pass

    def load_checkpoint(self):
        """
        Loads model state from checkpoint if it is provided
        """
        if not 'continue_from_iter' in self.config['firelab']: return

        self.num_iters_done = self.config['firelab'].get('continue_from_iter')
        self.num_epochs_done = self.num_iters_done // len(self.train_dataloader)

        for module_name in self.checkpoint_list:
            self.load_module_state(getattr(self, module_name), module_name)

    def should_stop(self):
        if self.max_num_iters and self.num_iters_done >= self.max_num_iters: return True
        if self.max_num_epochs and self.num_epochs_done >= self.max_num_epochs: return True
        if self.should_early_stop(): return True

        return False

    def should_early_stop(self):
        """Checks early stopping criteria"""
        return False

    def train_mode(self):
        """Switches all components into training mode"""
        pass

    def eval_mode(self):
        """Switches all components into evaluation mode"""
        pass

    def save_module_state(self, module, name):
        module_name = '{}-{}.pth'.format(name, self.num_iters_done)
        module_path = os.path.join(self.config['firelab']['checkpoints_path'], module_name)
        torch.save(module.state_dict(), module_path)

    def load_module_state(self, module, name, iter):
        module_name = '{}-{}.pth'.format(name, iter)
        module_path = os.path.join(self.config['firelab']['checkpoints_path'], module_name)
        module.load_state_dict(torch.load(module_path))
