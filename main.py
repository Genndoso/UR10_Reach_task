import argparse
import os

import pybullet as p
import yaml
from robot_descriptions.loaders.pybullet import load_robot_description

from src import ReacherEnv, ball, stable_bs_agent, record_video, DDPG_agent

parser = argparse.ArgumentParser(
    prog='RobotReacher',
)

parser.add_argument(
    '--logdir',
    type=str,
    default='logdir',
    help='Select logging directory')

parser.add_argument(
    '--model_dir',
    type=str,
    default='models',
    help='Select directory for saving models')

parser.add_argument(
    '--type_of_model',
    type=str,
  #  default='stable',
    help='Select type of model (from torch or stable_baselines')

parser.add_argument(
    '--train',
   # type=bool,
    #default=False,
    help='Train or test')

parser.add_argument(
    '--visualize',
    type=bool,
    default=True,
    help='Visualize environment')

parser.add_argument(
    '--config_path',
    type=str,
    default='config/sb_config.yaml',
    help='Select config')
    
parser.add_argument(
    '--path_to_model',
    type=str,
    default='best_model/best_model.zip',
    help='Select path to model') 
   

args = parser.parse_args()

with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)


def define_setup():
    p.connect(p.DIRECT)
    robot = load_robot_description("ur10_description", useFixedBase=True)
    tableUid = p.loadURDF("/src/objects/table/table.urdf", basePosition=[0.5, 0, -0.65])
    balls = ball(position=[0.5, -0.4, 0.4], radius=0.05)
    p.setGravity(0, 0, -10)
    p.setTimeStep(2)

    return robot, tableUid, balls


def main():
    """
    Training loop
    """
    
    robot, tableUid, balls = define_setup()
    env = ReacherEnv(robot)
    
    if 'stable' in args.type_of_model:
        agent = stable_bs_agent(args.type_of_model.split('_')[0], env, args.logdir, config)

    elif args.type_of_model == 'DDPG_torch':
        agent = DDPG_agent(env, config)
        
        
    print(args)
    
    
    if args.train:
    
        print('_________________________________________________')
        print('___________Starting training the agent___________')
        agent.learn()
        
        print('_________________________________________________')
        print('___________Agent is trained___________')

    else:
        agent.load_model(args.path_to_model)
        print('_________________________________________')
        print('___________Agent is loaded___________')

    if args.visualize:
        record_video(env, agent.model)
        print('______________________________')
        print('___________Video is recorded___________')


if __name__ == '__main__':
    main()
