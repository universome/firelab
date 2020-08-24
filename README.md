## firelab (version 0.0.20)
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

### Future plans
[ ] Run in daemon.
[ ] Implement `firelab ls` command
[ ] Easier profiling (via contexts?)
[ ] There are some interseting features in https://github.com/vpj/lab.
[ ] Add commit hash to summary
[ ] Create new branch/commit for each experiment?
[ ] More meaningful error messages.
[ ] Does model release GPU after training is finished (when we do not use HPO)?
[ ] Proper handling of errors in HPO: should we fail on the first exception? Should we try/catch result.get() in process pool?
[x] Make trainers run without config.firelab, this will make it possible to run trainer from python
[ ] Does continue_from_iter work?

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
- Why do we pass both exp_dir and exp_name everywhere in manager.py? We should care only about exp_path I suppose?
- Looks like we do not need the dublicating logic of directories creation in manager.py anymore since it is in BaseTrainer
- Rename BaseTrainer into Trainer?
