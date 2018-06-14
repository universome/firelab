from tqdm import tqdm#; tqdm.monitor_interval = 0


class BaseTrainer:
    def __init__(self, config):
        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs')
        self.max_num_iters = config.get('max_num_iters')
        self.validate_every = config.get('validate_every')
        self.plot_every = config.get('plot_every')
        self.early_stopping_last_n_iters = config.get('early_stopping_last_n_iters')
        self.log_file = config.get('log_file')
        self.should_validate = not self.validate_every is None
        self.should_plot = not self.plot_every is None

        self.train_dataloader = None

    def run_training(self):
        should_continue = True

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            try:
                for batch in tqdm(self.train_dataloader, leave=False):
                    self.train_on_batch(batch)
                    self.num_iters_done += 1
                    self.validate()
                    self.update_plots()
            except KeyboardInterrupt:
                should_continue = False
                break

            self.num_epochs_done += 1

    def train_on_batch(self, batch):
        pass

    def validate(self):
        pass

    def update_plots(self):
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
