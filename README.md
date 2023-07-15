# Universal robot control in pybullet environment with reinforcement learning
Universal robot control in pybullet environment

# Results 

Episode rewards             |  Actor loss              | Critic loss
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/Genndoso/UR10_Reach_task/blob/main/Media/Episode_reward.png))  |  ![](https://github.com/Genndoso/UR10_Reach_task/blob/main/Media/Actor_loss.png) | ![](https://github.com/Genndoso/UR10_Reach_task/blob/main/Media/Critic loss.png)


|Folder name       |                     Description                                    |
|------------------|--------------------------------------------------------------------|
|`best_model`   |  Best SACstable_baselines3 model                                         |
|`config`            | Config files for training stable_baselines3 and DDPG_torch models                               |
|`media`          | Media files                |
|`src`          |  source code             |

## Reward design


## Install
```
$ pip install git@github.com:Genndoso/UR10_Reach_task.git 

```

## Usage
```
Without docker

$ pip install -r requirements.txt
$ python3 main.py --logdir  --model_dir --type_of_model [stable, torch] --visualize [True, False] --train [True,False] --config_path --path_to_model
