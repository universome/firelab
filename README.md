## firelab (version 0.0.5)
### About
Framework for running DL experiments with pytorch.
Provides the following useful stuff:
- parallel hyperparameters optimization
- allows to `start`/`continue` your experiment with easy commands from yml config file
- easier to save checkpoints, write logs and visualize training
- useful utils for HP tuning and working with pytorch (look them up in `utils.py`)

### Installation
```
pip install firelab
```

### TODO
- daemon
- `firelab ls`
- easier profiling (via contexts?)
- there are some interseting features in https://github.com/vpj/lab
- add commit hash to summary
- create new branch/commit for each experiment?
- move HPO logic out from manager
- more meaningful error messages

### Useful commands:
- `firelab ls` — lists all running experiments
- `firelab start` / `firelab stop` / `firelab pause` / `firelab continue` — starts/stops/pauses/continues experiments

### Useful classes
- `BaseTrainer` — controls the experiment: loads data, runs/stops training, performs logging, etc

Cool staff firelab can do:
- Reduces amount of boilerplate code you write for training/running experiments
- Keep all experiment arguments and hyperparameters in a expressive config files
- Visualize your metrics with `tensorboard` through [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- Save checkpoints and logs with ease.
- Fixes random seeds for you by default (in numpy, pytorch and random). Attention: if you use other libs with other random generators, you should fix random seeds by yourself (we recommend taking it from hyperparams)

### Usage:
#### Configs
Besides your own configs, firelab adds its inner staff, which you can use or change as hyperparameter:
- `name` of the experiment
- `random_seed`

Experiment name determines where config is.
Experiment name can't duplicate each other.

### TODO
- Interactive config builder
- Clone experiment/config
- Add examples with several trainers in them
