import os
import pickle

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from firelab.utils.training_utils import is_history_improving, safe_oom_call, cudable


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs')
        self.max_num_iters = config.get('max_num_iters')
        self.losses = {}

        if self.config.get('checkpoint'):
            self.checkpoint_freq_iters = self.config.checkpoint.get('freq_iters')
            self.checkpoint_freq_epochs = self.config.checkpoint.get('freq_epochs')
            self.checkpoint_list = sum([self.config.modules.get(k) for k in self.config.modules.keys()], tuple())

            print('Will be checkpointing the following modules: {}'.format(self.checkpoint_list))

            if self.config.checkpoint.get('pickle'):
                assert type(self.config.checkpoint.pickle) is tuple
                print('Will be checkpointing with pickle the following modules: {}'.format(self.config.checkpoint.pickle))

            assert not (self.checkpoint_freq_iters and self.checkpoint_freq_epochs), """
                Can't save both on iters and epochs.
                Please, remove either freq_iters or freq_epochs
            """
        else:
            # TODO: govnokod :|
            self.checkpoint_freq_iters = None
            self.checkpoint_freq_epochs = None

        self.val_freq_iters = config.get('val_freq_iters')
        self.val_freq_epochs = config.get('val_freq_epochs')

        assert not(self.val_freq_iters and self.val_freq_epochs), """
            Can't validate on both iters and epochs.
            Please, remove either val_freq_iters or val_freq_epochs
        """

        self.train_dataloader = None
        self.val_dataloader = None

        self.writer = SummaryWriter(config.firelab.logs_path)

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
        try:
            while not self.should_stop():
                print('Running epoch #{}'.format(self.num_epochs_done+1))
                for batch in tqdm(self.train_dataloader):
                    batch = cudable(batch)

                    self.train_mode()
                    safe_oom_call(self.train_on_batch, batch, debug=self.config.get('debug_gpu'))

                    self.num_iters_done += 1

                    with torch.no_grad():
                        safe_oom_call(self.try_to_validate, debug=self.config.get('debug_gpu'))

                    self.checkpoint()
                self.num_epochs_done += 1
                self.on_epoch_done()
        except KeyboardInterrupt:
            print('\nTerminating experiment...')

    def train_on_batch(self, batch):
        pass

    def on_epoch_done(self):
        "Callback which is called when epoch has beed done"
        pass

    def try_to_validate(self):
        should_validate = False

        if self.val_freq_iters:
            should_validate = self.num_iters_done % self.val_freq_iters == 0
        elif self.val_freq_epochs:
            epoch_size = len(self.train_dataloader)
            was_epoch_just_finished = self.num_iters_done % epoch_size == 0
            is_epoch_appropriate = self.num_epochs_done % self.val_freq_epochs == 0
            should_validate = was_epoch_just_finished and is_epoch_appropriate

        if should_validate:
            self.eval_mode()
            self.validate()

    def validate(self):
        pass

    def checkpoint(self):
        should_checkpoint = False

        if self.checkpoint_freq_iters:
            should_checkpoint = self.num_iters_done % self.checkpoint_freq_iters == 0
        elif self.checkpoint_freq_epochs:
            # TODO: looks like govnokod
            epoch_size = len(self.train_dataloader)
            freq = self.checkpoint_freq_epochs * epoch_size
            should_checkpoint = self.num_iters_done % freq == 0

        if not should_checkpoint: return

        for module_name in self.checkpoint_list:
            self.save_module_state(getattr(self, module_name), module_name)

        if self.config.checkpoint.get('pickle'):
            for attr in self.config.checkpoint.pickle:
                self.pickle(getattr(self, attr), attr)

        self.checkpoint_freq_warning()

    def checkpoint_freq_warning(self):
        """
        Prints warning if we write checkpoints too often
        TODO: wip
        """
        pass

    def load_checkpoint(self):
        "Loads model state from checkpoint if it is provided"
        if self.config.firelab.get('continue_from_iter') is None: return

        if not self.config.firelab.reset_iters_counter:
            self.num_iters_done = self.config.firelab.continue_from_iter
            self.num_epochs_done = self.num_iters_done // len(self.train_dataloader)

        for module_name in self.checkpoint_list:
            self.load_module_state(getattr(self, module_name), module_name, self.config.firelab.continue_from_iter)

        if self.config.checkpoint.get('pickle'):
            for module_name in self.config.checkpoint.pickle:
                self.unpickle(module_name, self.config.firelab.continue_from_iter)

    def should_stop(self):
        "Checks all stopping criteria"
        if self.max_num_iters and self.num_iters_done >= self.max_num_iters:
            print('Terminating experiment because max num iters exceeded')
            return True

        if self.max_num_epochs and self.num_epochs_done >= self.max_num_epochs:
            print('Terminating experiment because max num epochs exceeded')
            return True

        if self.should_early_stop():
            print('Terminating experiment due to early stopping')
            return True

        return False

    def should_early_stop(self):
        "Checks early stopping criterion"
        if self.config.get('early_stopping') is None: return False

        history = self.losses[self.config.early_stopping.loss]
        n_steps = self.config.early_stopping.history_length
        should_decrease = self.config.early_stopping.should_decrease

        return not is_history_improving(history, n_steps, should_decrease)

    def train_mode(self):
        "Switches all models into training mode"
        for model_name in self.config.modules.models:
            getattr(self, model_name).train()

    def eval_mode(self):
        "Switches all models into evaluation mode"
        for model_name in self.config.modules.models:
            getattr(self, model_name).eval()

    def save_module_state(self, module, name):
        module_name = '{}-{}.pth'.format(name, self.num_iters_done)
        module_path = os.path.join(self.config.firelab.checkpoints_path, module_name)
        torch.save(module.state_dict(), module_path)

    def load_module_state(self, module, name, iteration):
        module_name = '{}-{}.pth'.format(name, iteration)
        module_path = os.path.join(self.config.firelab.checkpoints_path, module_name)
        module.load_state_dict(torch.load(module_path))

    def pickle(self, module, name):
        file_name = '{}-{}.pickle'.format(name, self.num_iters_done)
        path = os.path.join(self.config.firelab.checkpoints_path, file_name)
        pickle.dump(module, open(path, 'wb'))

    def unpickle(self, name, iteration):
        setattr(self, name, self.read_pickle_module(name, iteration))

    def read_pickle_module(self, name, iteration):
        file_name = '{}-{}.pickle'.format(name, iteration)
        path = os.path.join(self.config.firelab.checkpoints_path, file_name)

        return pickle.load(open(path, 'rb'))

    def write_losses(self, losses: dict, prefix=''):
        """
        Iterates over losses and logs them with self.writer
        Arguments:
            - losses: dict of losses; each loss should be a scalar
        """
        for k in losses:
            self.writer.add_scalar(prefix + k, losses[k], self.num_iters_done)
