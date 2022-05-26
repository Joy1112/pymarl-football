# Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning
This code is for reproducing the experiments results in **SPD: Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning**.
We custom the code from the Open-source code [pymarl2](https://github.com/hijkzzz/pymarl2).

## Requirements
* [PyTorch](https://pytorch.org/) >= 1.7.1
* python >= 3.8
* Google Research Football
* PettingZoo[mpe] == 1.17.0
* System: Ubuntu >= 18.04 is recommanded

## Installation
[Anaconda3](https://www.anaconda.com/) is recommanded for managing the experiment environment.
### Install Dependencies
```bash
# dependencies for gfootball
apt-get update && apt-get install git cmake build-essential \
    libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
```

### Install in the Anaconda Environment
```bash
# create conda environment
conda create -n spd python=3.8
conda activate spd

# install dependencies for gfootball
pip install --upgrade pip setuptools wheel
pip install psutil

# install gfootball
pip install ./third_party/football

# install mpe
pip install pettingzoo[mpe]==1.17.0

# install dependencies for pymarl2
pip install sacred numpy scipy gym matplotlib seaborn pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger pyvirtualdisplay tqdm protobuf==3.20.1

# install pytorch, please refer to the official site of PyTorch for a proper version.
# here we use torch==1.7.1+cu110 as an example.
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Run Experiments
We provide some neural networks parameters learned in our experiments in directory `results/models/`.
You can train new models following these instructions as well.
### MPE
#### URL training 
The config file is in the directory `src/config/algs/url_qmix_mpe.yaml` and you can check it for details about the hyper-parameters.
```bash
# SPD
python src/main.py --config=url_qmix_mpe --env-config=mpe_simple_tag with t_max=4050000

# DIAYN
python src/main.py --config=url_qmix_mpe --env-config=mpe_simple_tag with url_algo=diayn t_max=4050000

# WURL
python src/main.py --config=url_qmix_mpe --env-config=mpe_simple_tag with url_algo=wurl t_max=4050000
```
The learned policies will be in the folder `results/models/`, and named by the timestamp such as `gwd_qmix_simple_tag_agents-4__2022-05-12_15-52-17`.


#### URL evaluation
To reproduce the results in Sec. 5.1 in SPD, you need to specify the model location
```bash
# here we give an example
python src/main.py --config=url_eval_mpe --env-config=mpe_simple_tag --exp-config=results/models/gwd_qmix_simple_tag_agents-4__2022-05-12_15-52-17 with eval_process=True
```

### GRF
#### URL training
The config file is in the directory `src/config/algs/url_qmix_mpe.yaml` and you can check it for details about the hyper-parameters.
```bash
python src/main.py --config=url_gfootball --env-config=academy_3_vs_1_with_keeper with num_modes=20 ball_graph=True
```

#### Train on downstream tasks
```bash
# here we give an example. the details about env_args.map_style please refer to file `src/envs/gfootball/academy_3_vs_1_with_keeper.py`
python src/main.py --config=url_gfootball_load --env-config=academy_3_vs_1_with_keeper with num_modes=20 env_args.map_style=0 t_max=4050000 test_nepisode=50 epsilon_start=0.2 checkpoint_path=results/models/gwd_qmix_academy_3_vs_1_with_keeper_agents-3__2022-05-07_05-34-39
```
