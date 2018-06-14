from tqdm import tqdm; tqdm.monitor_interval = 0


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

    def run_training(self, training_data, val_data=None, plot_every=50, val_bleu_every=100):
        self.train_mode()

        self.val_data = val_data

        while not self.should_stop():
            try:
                for batch in tqdm(training_data, leave=False):
                    self.train_step(batch)
            except KeyboardInterrupt:
                self.is_interrupted = True
                break

            self.num_epochs_done += 1

    def train_step(self, batch):
        self.train_on_batch(batch)

        if self.should_validate and self.num_iters_done % self.validate_every == 0:
            self.validate(self.val_data)

        if self.should_plot and self.num_iters_done % self.plot_every == 0:
            self.plot_scores()

        self.num_iters_done += 1

    def should_stop(self):
        if self.max_num_iters and self.num_iters_done >= self.max_num_iters: return True
        if self.max_num_epochs and self.num_epochs_done >= self.max_num_epochs: return True
        if self.should_early_stop(): return True

        return False

    def should_early_stop(self):
        """Checks early stopping criteria"""
        return False

    def train_on_batch(self, batch):
        pass

    def validate(self, val_data):
        pass

    def plot_scores(self):
        pass

    def train_mode(self):
        pass

    def test_mode(self):
        pass
