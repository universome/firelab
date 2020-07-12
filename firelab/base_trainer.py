import os
import pickle
import logging
from typing import Dict, List, Callable
from datetime import timedelta

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import coloredlogs
from firelab.config import Config

from firelab.utils.training_utils import is_history_improving, safe_oom_call
from firelab.utils.fs_utils import infer_new_experiment_version
from firelab.config import Config

from .utils.distributed_utils import synchronize, is_main_process

class BaseTrainer:
    def __init__(self, config):
        # TODO: we should somehow say more loudly that we are reserving these properties
        # Besides, some properties are vital for user to define at he has not idea about it :|
        # TODO: even I do not know all the options available in config :|
        if config.has('base_config'):
            self.config = Config.load(config.base_config)
            self.config.overwrite(config)
        else:
            self.config = config

        self._init_logger()
        self._init_devices()

        if self.config.get('is_distributed', False):
            self.logger.info(f'Running on cuda:{self.gpus[0]}')
            torch.cuda.set_device(self.gpus[0])
            torch.distributed.init_process_group(
                backend=self.config.get('distributed_backend', 'nccl'),
                # Timeout for sync (works only for gloo)
                timeout=timedelta(0, self.config.get('distributed_timeout', 1800)))
            synchronize()

        self._init_paths()
        self._init_tb_writer()
        self._init_callbacks()
        self._init_checkpointing_strategy()
        self._init_validation_strategy()
        self._init_stopping_criteria()

        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.is_explicitly_stopped = False
        self.train_dataloader = None
        self.val_dataloader = None

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

    def get_training_results(self) -> Dict:
        """
        Function which returns training results which
        are passed to summary generation after training is done
        """
        return {}

    ######################
    ### Public methods ###
    ######################
    def start(self):
        if len(self.gpus) > 0:
            with torch.cuda.device(self.gpus[0]):
                self._start()
        else:
            self._start()

    def stop(self, stopping_reason:str=''):
        self.is_explicitly_stopped = True
        self._explicit_stopping_reason = stopping_reason

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
    def init(self):
        # Initialization
        self.before_init_hook()
        self.init_dataloaders()
        self.init_models()
        self.init_criterions()
        self.init_optimizers()
        self._try_to_load_checkpoint()
        self.after_init_hook()

    def _start(self):
        self.init()

        # Training
        self.before_training_hook()
        self._run_training()
        self.after_training_hook()
        self.writer.close()

    def _run_training(self):
        try:
            while not self._should_stop():
                if self.config.get('logging.training_progress', True) and is_main_process():
                    batches = tqdm(self.train_dataloader)

                    self.logger.info('Running epoch #{}'.format(self.num_epochs_done+1))
                else:
                    batches = self.train_dataloader

                for batch in batches:
                    self._set_train_mode()

                    if self.config.get('should_ignore_oom_batches', False):
                        safe_oom_call(self.train_on_batch, self.logger, batch, debug=self.config.get('debug_gpu'))
                    else:
                        self.train_on_batch(batch)

                    self.num_iters_done += 1

                    # Checkpointing the model BEFORE validation, since validation can hault :|
                    self._try_to_checkpoint()

                    if self.config.get('should_ignore_oom_batches', False):
                        safe_oom_call(self._try_to_validate, self.logger, debug=self.config.get('debug_gpu'))
                    else:
                        self._try_to_validate()

                    if self._should_stop():
                        break

                self.num_epochs_done += 1
                self.on_epoch_done()
        except Exception as e:
            self._terminate_experiment(str(e))
            raise

    def _try_to_validate(self):
        should_validate = False

        if self.val_freq_iters:
            should_validate = self.num_iters_done % self.val_freq_iters == 0
        elif self.val_freq_epochs:
            epoch_size = len(self.train_dataloader)
            was_epoch_just_finished = self.num_iters_done % epoch_size == 0
            # TODO: just use different callbacks for val_freq_epochs and val_freq_iters
            num_epochs_done = (self.num_epochs_done + 1) if was_epoch_just_finished else self.num_epochs_done
            is_epoch_appropriate = num_epochs_done % self.val_freq_epochs == 0
            should_validate = was_epoch_just_finished and is_epoch_appropriate

        if should_validate:
            self._set_eval_mode()

            # Validating without grad enabled (less memory consumption)
            with torch.no_grad():
                self.validate()

    def _try_to_checkpoint(self):
        # Checkpointing in non-main processes lead to subtle erros when loading the weights
        if not is_main_process(): return

        should_checkpoint = False

        if self.checkpoint_freq_iters:
            should_checkpoint = self.num_iters_done % self.checkpoint_freq_iters == 0
        elif self.checkpoint_freq_epochs:
            # TODO: looks like govnokod
            epoch_size = len(self.train_dataloader)
            freq = self.checkpoint_freq_epochs * epoch_size
            should_checkpoint = self.num_iters_done % freq == 0

        if not should_checkpoint:
            return

        self.checkpoint()
        self._checkpoint_freq_warning()

    def checkpoint(self):
        # We want to checkpoint right now!
        if not self.paths.has('checkpoints_path'):
            raise RuntimeError(
                'Tried to checkpoint, but no checkpoint path was specified. Cannot checkpoint.'\
                'Provide either `paths.checkpoints_path` or `experiment_dir` in config.')

        for module_name in self.config.get('checkpoint.modules', []):
            self._save_module_state(getattr(self, module_name), module_name)

        for pickle_attr in self.config.get('checkpoint.pickle', []):
            self._pickle(getattr(self, pickle_attr), pickle_attr)

    def _checkpoint_freq_warning(self):
        """
        Prints warning if we write checkpoints too often
        TODO: wip
        """
        pass

    def _try_to_load_checkpoint(self):
        "Loads model state from checkpoint if it is provided"
        if not self.config.has('continue_from_iter'): return

        if not self.config.reset_iters_counter:
            self.num_iters_done = self.config.continue_from_iter
            self.num_epochs_done = self.num_iters_done // len(self.train_dataloader)

        for module_name in self.config.checkpoint.modules:
            self._load_module_state(getattr(self, module_name), module_name, self.config.continue_from_iter)

        for module_name in self.config.get('checkpoint.pickle', []):
            self._unpickle(module_name, self.config.continue_from_iter)

    def _should_stop(self) -> bool:
        "Checks all stopping criteria"
        if (not self.max_num_iters is None) and (self.num_iters_done >= self.max_num_iters):
            self._terminate_experiment('Max num iters exceeded')
            return True

        if (not self.max_num_epochs is None) and (self.num_epochs_done >= self.max_num_epochs):
            self._terminate_experiment('Max num epochs exceeded')
            return True

        if self._should_early_stop():
            self._terminate_experiment('Early stopping')
            return True

        if self.is_explicitly_stopped:
            self._terminate_experiment(f'Stopped explicitly via .stop() method. Reason: {self._explicit_stopping_reason}')
            return True

        return False

    def _should_early_stop(self):
        "Checks early stopping criterion"
        if self.config.get('early_stopping') is None: return False

        history = self.losses[self.config.early_stopping.loss_name]
        n_steps = self.config.early_stopping.history_length
        should_decrease = self.config.early_stopping.should_decrease

        return not is_history_improving(history, n_steps, should_decrease)

    # TODO: we can gather modules automaticall (via "isinstance")
    def _set_train_mode(self, flag: bool=True):
        """Switches all models into training mode"""

        for model_name in self.config.get('modules.models', []):
            getattr(self, model_name).train(flag)

    def _set_eval_mode(self):
        "Switches all models into evaluation mode"
        self._set_train_mode(False)

    def _save_module_state(self, module, name):
        module_name = '{}-{}.pt'.format(name, self.num_iters_done)
        module_path = os.path.join(self.paths.checkpoints_path, module_name)
        torch.save(module.state_dict(), module_path)

    def _load_module_state(self, module, name, iteration):
        module_name = '{}-{}.pt'.format(name, iteration)
        module_path = os.path.join(self.paths.checkpoints_path, module_name)
        module.load_state_dict(torch.load(module_path))

    def _pickle(self, module, name):
        file_name = '{}-{}.pickle'.format(name, self.num_iters_done)
        path = os.path.join(self.paths.checkpoints_path, file_name)
        pickle.dump(module, open(path, 'wb'))

    def _unpickle(self, name, iteration):
        setattr(self, name, self._read_pickle_module(name, iteration))

    def _read_pickle_module(self, name, iteration):
        file_name = '{}-{}.pickle'.format(name, iteration)
        path = os.path.join(self.paths.checkpoints_path, file_name)

        return pickle.load(open(path, 'rb'))

    def _terminate_experiment(self, termination_reason):
        self.logger.info('Terminating experiment because [%s]' % termination_reason)
        self._write_summary(termination_reason)

    def _write_summary(self, termination_reason:str):
        if not self.paths.has('summary_path'): return

        summary = {
            'name': self.config.get('exp_name', 'unnamed'),
            'termination_reason': termination_reason,
            'num_iters_done': self.num_iters_done,
            'num_epochs_done': self.num_epochs_done,
            'config': self.config.to_dict(),
            'results': self.get_training_results()
        }

        with open(self.paths.summary_path, 'w') as f:
            yaml.safe_dump(summary, f, default_flow_style=False)

    ##############################
    ### Initialization methods ###
    ##############################
    def _init_logger(self):
        if self.config.has('exp_name'):
            self.logger = logging.getLogger(self.config.exp_name)
        else:
            # TODO: is it okay to use class name?
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.warn('You should provide experiment name (by setting "exp_name" attribute in config) ' \
                             'if you want trainer logger to have a specific name.')

        coloredlogs.install(level=self.config.get('logging.level', 'DEBUG'), logger=self.logger)

    def _init_paths(self):
        if self.config.has('firelab.experiment_dir'):
            self.create_paths(self.config.firelab.experiment_dir)
        elif self.config.has('experiment_dir'):
            # We are only given a path to experiment dir. Have to create all the paths by ourselves
            self.create_paths(experiment_dir)
        else:
            # Saving into `experiments` directory
            os.makedirs('experiments', exist_ok=True)
            exp_base_name = self.config.get('exp_name', 'unnamed-experiment')
            exp_version = infer_new_experiment_version('experiments', exp_base_name)
            experiment_dir = os.path.join('experiments', f'{exp_base_name}-{exp_version:05d}')

            self.create_paths(experiment_dir)

    def create_paths(self, experiment_dir: str) -> Config:
        self.logger.info(f'Will be saving checkpoints/logs/etc into {experiment_dir} directory.')

        self.paths = Config({
            'experiment_dir': experiment_dir,
            'checkpoints_path': os.path.join(experiment_dir, 'checkpoints'),
            'summary_path': os.path.join(experiment_dir, 'summary.yml'),
            'config_path': os.path.join(experiment_dir, 'config.yml'),
            'logs_path': os.path.join(experiment_dir, 'logs'),
            'custom_data_path': os.path.join(experiment_dir, 'custom_data'),
        })

        if is_main_process():
            os.makedirs(self.paths.experiment_dir)
            os.makedirs(self.paths.checkpoints_path)
            os.makedirs(self.paths.logs_path)
            os.makedirs(self.paths.custom_data_path)
            os.makedirs(os.path.dirname(self.paths.summary_path), exist_ok=True)

            self.config.save(self.paths.config_path)

    def _init_tb_writer(self):
        if not self.paths.has('logs_path') or not is_main_process():
            logger = self.logger

            # TODO: maybe we should just raise an exception?
            class DummyWriter:
                def __getattribute__(self, name):
                    dummy_fn = lambda *args, **kwargs: None
                    logger.warn('Tried to use tensorboard, but tensorboard logs dir was not set. Nothing is written.')
                    return dummy_fn

            self.writer = DummyWriter()
        else:
            self.writer = SummaryWriter(
                self.paths.logs_path,
                flush_secs=self.config.get('logging.tb_flush_secs', 5))

    def _init_callbacks(self):
        self._on_iter_done_callbacks: List[Callable] = []
        self._on_epoch_done_callbacks: List[Callable] = []
        self._on_training_done_callbacks: List[Callable] = []

    def _init_checkpointing_strategy(self):
        if self.config.get('checkpoint'):
            self.checkpoint_freq_iters = self.config.checkpoint.get('freq_iters')
            self.checkpoint_freq_epochs = self.config.checkpoint.get('freq_epochs')

            if len(self.config.get('checkpoint.modules')):
                self.logger.warn(
                    '`checkpoint` config is specified, but no `modules` are provided. '
                    'No torch modules to checkpoint!')

            if self.config.checkpoint.get('pickle'):
                assert type(self.config.checkpoint.pickle) is tuple
                self.logger.info(
                    f'Will be checkpointing with pickle' \
                    f'the following modules: {self.config.checkpoint.pickle}')

            assert not (self.checkpoint_freq_iters and self.checkpoint_freq_epochs), """
                Can't save both on iters and epochs.
                Please, remove either freq_iters or freq_epochs
            """
        else:
            # TODO: govnokod :|
            self.checkpoint_freq_iters = None
            self.checkpoint_freq_epochs = None

    def _init_validation_strategy(self):
        self.val_freq_iters = self.config.get('val_freq_iters')
        self.val_freq_epochs = self.config.get('val_freq_epochs')

        assert not(self.val_freq_iters and self.val_freq_epochs), """
            Can't validate on both iters and epochs.
            Please, remove either val_freq_iters or val_freq_epochs
        """

    def _init_stopping_criteria(self):
        self.max_num_epochs = self.config.get('hp.max_num_epochs')
        self.max_num_iters = self.config.get('hp.max_num_iters')
        self.losses = {}

        if not (self.max_num_iters or self.max_num_epochs or self.config.has('early_stopping')):
            raise ValueError('You should set either `max_num_iters` or `max_num_epochs`')

    def _init_devices(self):
        assert not self.config.has('device_name'), \
            'FireLab detects and sets `device_name` for you. You influence it via `gpus`.'
        assert not hasattr(self, 'device_name'), 'You should not overwrite "device_name" attribute in Trainer.'
        assert not hasattr(self, 'gpus'), 'You should not overwrite "gpus" attribute in Trainer.'

        visible_gpus = list(range(torch.cuda.device_count()))

        if self.config.has('gpus'):
            self.gpus = self.config.gpus
        elif self.config.has('firelab.gpus'):
            self.gpus = self.config.firelab.gpus
        else:
            # TODO: maybe we should better take GPUs only when allowed?
            self.gpus = visible_gpus
            self.logger.warn(f'Attribute "gpus" was not set in config and '
                             f'{len(visible_gpus)} GPUs were found. I gonna use them all.')

        if len(self.gpus) > 0:
            self.device_name = f'cuda:{self.gpus[0]}'
        else:
            self.device_name = 'cpu'
