### About
Framework for running DL experiments with pytorch
Provides the following helping stuff:
- `Runner` — controls the experiment: loads data, runs/stops training, etc
- `Trainer` — trains your model (runs iterations)
- `Plotter` — runs on top of logger. To plot smth you have to log it first (this is quite obvious: we cant plot stuff without reference to it. And keeping everything in memory is not a good idea)
- `hp` — helps configuring hyperparameters.
- `configs` — configs for experiments (WIP)


### Useful commands:
- `firelab ls` — lists all running experiments
- `firelab start` / `firelab stop` / `firelab pause` / `firelab continue` — starts/stops/pauses/continues experiments

Cool staff firelab can do:
- Out-of-the-box integration with `visdom`
- Does not let you start experiments with the same configs
- Fixes random seeds for you by default (in numpy, pytorch and random). Attention: if you use other libs with other random generators, you should fix random seeds by yourself (we recommend taking it from hyperparams)

### Usage:
#### Logger
```
import firelab

logger = new firelab.Logger()
logger.log('bleu_loss', 0.8, num_iters_done=exp.num_iters_done())
logger.log('mse_loss', 0.8, num_iters_done=exp.num_iters_done())
logger.log_many('bleu_loss', num_iters_done=exp.num_iters_done())
```

#### Configs
Besides your own configs, firelab adds its inner staff, which you can use or change as hyperparameter:
- experiment name
- random seed

Experiment name determines where config is.
Experiment name can't duplicate each other.

```
from firelab import experiment

exp = Experiment()

exp.run()
# TODO: exp.run_in_jupyter(log_every_n_iters=10, log_every_n_epochs=100) — it will additionally plot

exp.config.get('max_num_iters')
```

### TODO
- Interactive config builder
- Clone experiment/config
