# Universal robot control in pybullet environment with reinforcement learning
Universal robot control in pybullet environment. Maximum about 200k timesteps were conducted. Because of lack of computational power I couldn't train the agent.

# Project structure

|Folder name       |                     Description                                    |
|------------------|--------------------------------------------------------------------|
|`best_model`   |  Best SACstable_baselines3 model                                         |
|`config`            | Config files for training stable_baselines3 and DDPG_torch models                               |
|`media`          | Media files                |
|`src`          |  source code             |

# Results 

[![Rendering](https://github.com/Genndoso/UR10_Reach_task/blob/main/Media/vid.mp4)

Episode rewards             |  Actor loss              | Critic loss
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/Genndoso/UR10_Reach_task/blob/main/Media/eval_mean_reward.png)  |  ![](https://github.com/Genndoso/UR10_Reach_task/blob/main/Media/Actor_loss.png) | ![](https://github.com/Genndoso/UR10_Reach_task/blob/main/Media/Critic_loss.png)


## Reward design
Reward is based on negative L1 distance between end joint of UR10 robot and target position plus reward for reaching the target position minus constraint for collission with table.

$$ -\sum_{k=1}^3  joint_k - tp_k   + gr - collission $$


## Install
```
$ pip install git@github.com:Genndoso/UR10_Reach_task.git 

```

## Usage
```
Without docker

$ pip install -r requirements.txt
$ python3 main.py --logdir  --model_dir --type_of_model [stable, torch] --visualize [True, False] --train [True,False] --config_path --path_to_model

With docker (ex)

$ docker build -t $your_image_name$ --build-arg type_of_model=DDPG_torch --build-arg config_path=config/config_torch.yaml --build-arg train=True --build-arg visualize=True .

