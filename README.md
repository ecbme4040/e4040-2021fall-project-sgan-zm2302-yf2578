# e4040-2021Fall-SGAN-zm2302-yf2578
This is a project to reproduce the paper:
**<a href="https://arxiv.org/abs/1803.10892">Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks</a>**
<br>
<a href="http://web.stanford.edu/~agrim/">Agrim Gupta</a>,
<a href="http://cs.stanford.edu/people/jcjohns/">Justin Johnson</a>,
<a href="http://vision.stanford.edu/feifeili/">Fei-Fei Li</a>,
<a href="http://cvgl.stanford.edu/silvio/">Silvio Savarese</a>,
<a href="http://web.stanford.edu/~alahi/">Alexandre Alahi</a>
<br>
Presented at [CVPR 2018](http://cvpr2018.thecvf.com/)

In this paper, pedestrain trajectory is predicted using the social GAN model. The main contribution of this paper is that
it proposes "social-pooling" technique to consider neighbouring pedestrains when making prediction.

We make two modifications:
1. We change the application field from pedestrian trajectory prediction to human driving behavior prediction, which is a car-following modeling problem.
2. We partially implemented the GAN model without “social pooling” due to its complexity.

The reason for the 1st modification is because the original paper is too complex, which contains the GAN structure and “social-pooling”, and we focus on the GAN structure,
which is the main framework of the original paper. The reason for the 2nd modification is that we are with a transportation background and want to try to apply this method to our domain. 

We aims to use the GAN model to predict the car-following behavior considering human uncertainty, and the result is shown as below:
<div align='center'>
  <img src='image/mode=paper-sudoku-level=3-lowest_kl=1.png' width='500px'>
</div>

In the figure, the x-axis is the target velocity and the y-axis is its probability density. The red and blue curves are the results of the ground-truth and prediction, respectively. We can see that GAN model can fit the 
distribution of the car-following behavior very well.
# Code
This repo has two characteristics:
1. All procedures (training, testing, visualization) can be run in the terminal by scripts. Thus the notebook is very clean.
2. The result is "folder-based". Each folder in the experiments means contains all files related to this folder, including the configuration file, model weights,
test results, and visualization.

# How to reproduce

There are only 3 cells in the notebook that are enough to reproduce the results. The merit goes to the aforementioned 1st characteristic that all moudules are in script.
Instead of running the notebook, you can also run the following codes:
```bash
python main_lstm.py --experiment_dir experiments/ngsim/gan --data_dir data/ngsim/ngsim_v_dt=1 --mode train --force_overwrite
python main_lstm.py --experiment_dir experiments/ngsim/gan --data_dir data/ngsim/ngsim_v_dt=1 --mode test
python viz.py --experiment_dir experiments/ngsim/gan --sudoku --interval --metrics_statistic --force_overwrite
```

# Code architecture
#### rood directory
 - data: a folder containing processed data
 - experiments: folders to save everything related to one training (or one experiment), including 
 - image: images for README.md
 - raw_data: a folder containing raw NGSIM trajectory data
 - src: everything related to the model structure, including GAN model, trianing and test scripts.
 - tmp: temporary folder
 - build_idm_data.py: generate numerical data from the intelligent driving model. Not used in this case study where we use the 
real-world NGSIM dataset
 - build_ngsim_data.py: process NGSIM data
 - main_lstm.py: main script
 - viz.py: code for visualization
 - E4040.2021Fall.SGAN.report.zm2302.yf2578.pdf: report

#### src
- layers: class for the generator, discriminator, poster estimator, and neural networks
- metrics_factory: functions to calculate the KL divergence and NLPD
- models: contains GAN model and NN-Drop model, and relevant functions to calculate the loss functions
- argparse.py: to parse the argument
- helper_funcs.py: functions like save and load
- metrics.py: map the names of metrics to the metrics instances
- picklefuncs.py: tool functions
- training.py: tranining and test procedures.
- utils.py: tool functions

# Dataset
Raw dataset is stored in the folder "raw_data/NGSIM". It is a pickle file contains the trajectory information of the follower and the leader.
To convert the time-series trajectory to feature-label pairs, "build_ngsim_data.py" should be run:

```bash
python build_ngsim_data.py
```
This script will first load the "data_para.json" file that contains the data configuration like the training and test total size. Then the process data will
be save in the .csv file in the same location as the json file.

This step can be skipped as the processed data is in the repo.

# Model
We uses the keras subclass to customize our own model. Specifically, we code the single networks (e.g. generator) as
keras layers, and code the combination of networks (e.g. generator+discriminator) as keras models.

# Organization of this directory
To be populated by students, as shown in previous assignments.
Create a directory/file tree

```./
├── build_idm_data.py
├── build_ngsim_data.py
├── data
│   └── ngsim
│       ├── ngsim_v_dt=01
│       │   ├── collocation_feature.csv
│       │   ├── data_para.json
│       │   ├── test_feature.csv
│       │   ├── test_label.csv
│       │   ├── train_feature.csv
│       │   ├── train_label.csv
│       │   ├── validation_feature.csv
│       │   └── validation_label.csv
│       ├── ngsim_v_dt=1
│       │   ├── collocation_feature.csv
│       │   ├── data_para.json
│       │   ├── test_feature.csv
│       │   ├── test_label.csv
│       │   ├── train_feature.csv
│       │   ├── train_label.csv
│       │   ├── validation_feature.csv
│       │   └── validation_label.csv
│       └── ngsim_v_dt=5
│           ├── collocation_feature.csv
│           ├── data_para.json
│           ├── test_feature.csv
│           ├── test_label.csv
│           ├── train_feature.csv
│           ├── train_label.csv
│           ├── validation_feature.csv
│           └── validation_label.csv
├── experiments
│   └── ngsim
│       ├── gan
│       │   ├── best_weights
│       │   │   ├── after-epoch-739.data-00000-of-00001
│       │   │   ├── after-epoch-739.index
│       │   │   └── checkpoint
│       │   ├── experiment_setting.json
│       │   ├── last_weights
│       │   │   ├── after-epoch-1000.data-00000-of-00001
│       │   │   ├── after-epoch-1000.index
│       │   │   └── checkpoint
│       │   ├── metrics_eval_best_weights.json
│       │   ├── metrics_eval_last_weights.json
│       │   ├── physics_last_weights
│       │   ├── summary
│       │   │   ├── eval
│       │   │   │   └── events.out.tfevents.1640146605.zhaobinpc.19317.1.v2
│       │   │   └── train
│       │   │       └── events.out.tfevents.1640146605.zhaobinpc.19317.0.v2
│       │   ├── test
│       │   │   ├── best_weights
│       │   │   │   ├── features_test.csv
│       │   │   │   ├── kl_test.csv
│       │   │   │   ├── metrics_test.json
│       │   │   │   ├── predictions_test.csv
│       │   │   │   └── targets_test.csv
│       │   │   └── last_weights
│       │   │       ├── features_test.csv
│       │   │       ├── kl_test.csv
│       │   │       ├── metrics_test.json
│       │   │       ├── predictions_test.csv
│       │   │       └── targets_test.csv
│       │   ├── train.log
│       │   └── viz
│       │       ├── best_weights
│       │       │   ├── metrics_statistic.json
│       │       │   ├── mode=debug-sudoku-level=3-lowest_kl=0.png
│       │       │   ├── mode=debug-sudoku-level=3-lowest_kl=1.png
│       │       │   ├── mode=debug-type=confidence-sorted=True-bounds=False.png
│       │       │   ├── mode=debug-type=prediction-sorted=True-bounds=False.png
│       │       │   ├── mode=paper-sudoku-level=3-lowest_kl=0.png
│       │       │   ├── mode=paper-sudoku-level=3-lowest_kl=1.png
│       │       │   ├── mode=paper-type=confidence-sorted=True-bounds=False.png
│       │       │   └── mode=paper-type=prediction-sorted=True-bounds=False.png
│       │       └── last_weights
│       │           ├── metrics_statistic.json
│       │           ├── mode=debug-sudoku-level=3-lowest_kl=0.png
│       │           ├── mode=debug-sudoku-level=3-lowest_kl=1.png
│       │           ├── mode=debug-type=confidence-sorted=True-bounds=False.png
│       │           ├── mode=debug-type=prediction-sorted=True-bounds=False.png
│       │           ├── mode=paper-sudoku-level=3-lowest_kl=0.png
│       │           ├── mode=paper-sudoku-level=3-lowest_kl=1.png
│       │           ├── mode=paper-type=confidence-sorted=True-bounds=False.png
│       │           └── mode=paper-type=prediction-sorted=True-bounds=False.png
│       └── nn_drop
│           ├── experiment_setting.json
│           └── train.log
├── image
│   └── mode=paper-sudoku-level=3-lowest_kl=1.png
├── main.ipynb
├── main_lstm.py
├── raw_data
│   └── ngsim
│       └── IDOVM_para_and_mid_and_73weight_paraspace_cluster.pickle
├── README.md
├── report.pdf
├── src
│   ├── argparse.py
│   ├── helper_funcs.py
│   ├── __init__.py
│   ├── layers
│   │   ├── gan.py
│   │   ├── __init__.py
│   │   ├── nn_drop.py
│   │   ├── __pycache__
│   │   │   ├── gan.cpython-36.pyc
│   │   │   ├── gan.cpython-37.pyc
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   ├── nn_drop.cpython-36.pyc
│   │   │   ├── nn_drop.cpython-37.pyc
│   │   │   ├── utils.cpython-36.pyc
│   │   │   └── utils.cpython-37.pyc
│   │   └── utils.py
│   ├── metrics_factory
│   │   ├── get_KL.py
│   │   ├── get_NLPD.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── get_KL.cpython-36.pyc
│   │       ├── get_KL.cpython-37.pyc
│   │       ├── get_NLPD.cpython-36.pyc
│   │       ├── get_NLPD.cpython-37.pyc
│   │       ├── __init__.cpython-36.pyc
│   │       └── __init__.cpython-37.pyc
│   ├── metrics.py
│   ├── models
│   │   ├── gan.py
│   │   ├── __init__.py
│   │   ├── loss_calculation.py
│   │   ├── nn_drop.py
│   │   └── __pycache__
│   │       ├── bsc_physgan.cpython-37.pyc
│   │       ├── double_physgan.cpython-37.pyc
│   │       ├── gan.cpython-36.pyc
│   │       ├── gan.cpython-37.pyc
│   │       ├── __init__.cpython-36.pyc
│   │       ├── __init__.cpython-37.pyc
│   │       ├── loss_calculation.cpython-36.pyc
│   │       ├── loss_calculation.cpython-37.pyc
│   │       ├── mmt_physgan.cpython-37.pyc
│   │       ├── nn_drop.cpython-36.pyc
│   │       ├── nn_drop.cpython-37.pyc
│   │       └── pinn_drop.cpython-37.pyc
│   ├── picklefuncs.py
│   ├── __pycache__
│   │   ├── argparse.cpython-37.pyc
│   │   ├── helper_funcs.cpython-36.pyc
│   │   ├── helper_funcs.cpython-37.pyc
│   │   ├── __init__.cpython-36.pyc
│   │   ├── __init__.cpython-37.pyc
│   │   ├── metrics.cpython-36.pyc
│   │   ├── metrics.cpython-37.pyc
│   │   ├── training.cpython-36.pyc
│   │   ├── training.cpython-37.pyc
│   │   ├── utils.cpython-36.pyc
│   │   └── utils.cpython-37.pyc
│   ├── training.py
│   └── utils.py
├── tmp
│   └── tmp
└── viz.py

34 directories, 130 files
