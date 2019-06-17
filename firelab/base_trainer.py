import os
import pickle
import logging

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import coloredlogs


from firelab.utils.training_utils import is_history_improving, safe_oom_call


class BaseTrainer:
    def __init__(self, config):
        # TODO: we should somehow say more loudly that we are reserving these properties
        # Besides, some properties are vital for user to define at he has not idea about it :|
        self.config = config
        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs')
        self.max_num_iters = config.get('max_num_iters')
        self.losses = {}
        self.logger = logging.getLogger(self.config.firelab.exp_name)
        self.is_explicitly_stopped = False

        coloredlogs.install(level=self.config.get('logging.level', 'DEBUG'), logger=self.logger)

        if self.config.get('checkpoint'):
            self.checkpoint_freq_iters = self.config.checkpoint.get('freq_iters')
            self.checkpoint_freq_epochs = self.config.checkpoint.get('freq_epochs')
            self.checkpoint_list = sum([self.config.modules.get(k) for k in self.config.modules.keys()], tuple())

            self.logger.info('Will be checkpointing the following modules: {}'.format(self.checkpoint_list))

            if self.config.checkpoint.get('pickle'):
                assert type(self.config.checkpoint.pickle) is tuple
                self.logger.info('Will be checkpointing with pickle the following modules: {}'.format(self.config.checkpoint.pickle))

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

        self.writer = SummaryWriter(config.firelab.logs_path, flush_secs=5)

        if not (self.max_num_iters or self.max_num_epochs or self.config.has('early_stopping')):
            self.logger.warn('You did not specify any stopping criteria (max_num_iters, max_num_epochs, early_stopping). I am going to run forever. Huehuehue.')

    ############################
    ### Overwritable methods ###
    ############################
    def init_dataloaders(self):
        pass

    def init_models(self):
        pass

    def init_criterions(self):
        pass

    def init_optimizers(self):
        pass

    def train_on_batch(self, batch):
        pass

    def on_epoch_done(self):
        "Callback which is called when epoch has beed done"
        pass

    def validate(self):
        pass

    #############
    ### Hooks ###
    #############
    def before_init_hook(self):
        pass

    def after_init_hook(self):
        pass

    def before_training_hook(self):
        pass

    def after_training_hook(self):
        pass

    ######################
    ### Public methods ###
    ######################
    def start(self):
        if len(self.config.firelab.available_gpus) > 0:
            with torch.cuda.device(self.config.firelab.available_gpus[0]):
                self._start()
        else:
            self._start()

    def stop(self):
        self.is_explicitly_stopped = True

    def write_losses(self, losses: dict, prefix=''):
        """
        Iterates over losses and logs them with self.writer
        Arguments:
            - losses: dict of losses; each loss should be a scalar
        """
        for k in losses:
            self.writer.add_scalar(prefix + k, losses[k], self.num_iters_done)

    #######################
    ### Private methods ###
    #######################
    def _start(self):
        # Initialization
        self.before_init_hook()
        self.init_dataloaders()
        self.init_models()
        self.init_criterions()
        self.init_optimizers()
        self._try_to_load_checkpoint()
        self.after_init_hook()

        # Training
        self.before_training_hook()
        self._run_training()
        self.after_training_hook()
        self.writer.close()

    def _run_training(self):
        try:
            while not self._should_stop():
                if self.config.get('logging.training_progress', True):
                    batches = tqdm(self.train_dataloader)

                    self.logger.info('Running epoch #{}'.format(self.num_epochs_done+1))
                else:
                    batches = self.train_dataloader

                for batch in batches:
                    self._set_train_mode()
                    safe_oom_call(self.train_on_batch, self.logger, batch, debug=self.config.get('debug_gpu'))

                    self.num_iters_done += 1

                    # Let's validate without grad enabled (less memory consumption)
                    with torch.no_grad():
                        safe_oom_call(self._try_to_validate, self.logger, debug=self.config.get('debug_gpu'))

                    self._checkpoint()

                    if self._should_stop():
                        break

                self.num_epochs_done += 1
                self.on_epoch_done()
        except Exception as e:
            self._write_summary(str(e))
            raise

    def _try_to_validate(self):
        should_validate = False

        if self.val_freq_iters:
            should_validate = self.num_iters_done % self.val_freq_iters == 0
        elif self.val_freq_epochs:
            epoch_size = len(self.train_dataloader)
            was_epoch_just_finished = self.num_iters_done % epoch_size == 0
            is_epoch_appropriate = self.num_epochs_done % self.val_freq_epochs == 0
            should_validate = was_epoch_just_finished and is_epoch_appropriate

        if should_validate:
            self._set_eval_mode()
            self.validate()


    def _checkpoint(self):
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
            self._save_module_state(getattr(self, module_name), module_name)

        if self.config.checkpoint.get('pickle'):
            for attr in self.config.checkpoint.pickle:
                self._pickle(getattr(self, attr), attr)

        self._checkpoint_freq_warning()

    def _checkpoint_freq_warning(self):
        """
        Prints warning if we write checkpoints too often
        TODO: wip
        """
        pass

    def _try_to_load_checkpoint(self):
        "Loads model state from checkpoint if it is provided"
        if self.config.firelab.get('continue_from_iter') is None: return

        if not self.config.firelab.reset_iters_counter:
            self.num_iters_done = self.config.firelab.continue_from_iter
            self.num_epochs_done = self.num_iters_done // len(self.train_dataloader)

        for module_name in self.checkpoint_list:
            self._load_module_state(getattr(self, module_name), module_name, self.config.firelab.continue_from_iter)

        if self.config.checkpoint.get('pickle'):
            for module_name in self.config.checkpoint.pickle:
                self._unpickle(module_name, self.config.firelab.continue_from_iter)

    def _should_stop(self) -> bool:
        "Checks all stopping criteria"
        if (not self.max_num_iters is None) and (self.num_iters_done >= self.max_num_iters):
            self._write_summary('Max num iters exceeded')
            return True

        if (not self.max_num_epochs is None) and (self.num_epochs_done >= self.max_num_epochs):
            self._write_summary('Max num epochs exceeded')
            return True

        if self._should_early_stop():
            self._write_summary('Early stopping')
            return True

        if self.is_explicitly_stopped:
            self._write_summary('Stopped explicitly via .stop() method')
            return True

        return False

    def _should_early_stop(self):
        "Checks early stopping criterion"
        if self.config.get('early_stopping') is None: return False

        history = self.losses[self.config.early_stopping.loss]
        n_steps = self.config.early_stopping.history_length
        should_decrease = self.config.early_stopping.should_decrease

        return not is_history_improving(history, n_steps, should_decrease)

    # TODO: we can gather modules automaticall (via "isinstance")
    def _set_train_mode(self, flag:bool=True):
        "Switches all models into training mode"

        if not self.config.has('modules'): return
        if not self.config.modules.has('models'): return

        for model_name in self.config.modules.models:
            getattr(self, model_name).train(flag)

    def _set_eval_mode(self):
        "Switches all models into evaluation mode"
        self._set_train_mode(False)

    def _save_module_state(self, module, name):
        module_name = '{}-{}.pt'.format(name, self.num_iters_done)
        module_path = os.path.join(self.config.firelab.checkpoints_path, module_name)
        torch.save(module.state_dict(), module_path)

    def _load_module_state(self, module, name, iteration):
        module_name = '{}-{}.pt'.format(name, iteration)
        module_path = os.path.join(self.config.firelab.checkpoints_path, module_name)
        module.load_state_dict(torch.load(module_path))

    def _pickle(self, module, name):
        file_name = '{}-{}.pickle'.format(name, self.num_iters_done)
        path = os.path.join(self.config.firelab.checkpoints_path, file_name)
        pickle.dump(module, open(path, 'wb'))

    def _unpickle(self, name, iteration):
        setattr(self, name, self._read_pickle_module(name, iteration))

    def _read_pickle_module(self, name, iteration):
        file_name = '{}-{}.pickle'.format(name, iteration)
        path = os.path.join(self.config.firelab.checkpoints_path, file_name)

        return pickle.load(open(path, 'rb'))

    def _write_summary(self, termination_reason:str):
        self.logger.info('Terminating experiment because [%s]' % termination_reason)

        summary = {
            'name': self.config.firelab.exp_name,
            'termination_reason': termination_reason,
            'num_iters_done': self.num_iters_done,
            'num_epochs_done': self.num_epochs_done,
            'config': self.config.to_dict()
        }

        with open(self.config.firelab.summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
