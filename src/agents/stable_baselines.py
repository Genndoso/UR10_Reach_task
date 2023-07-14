from stable_baselines3 import SAC, DDPG, PPO
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class stable_bs_agent():
    def __init__(self, algorithm, env, logdir, config):
        self.config = config
        self.env = env
        self.algorithm = algorithm

        if algorithm == 'PPO':
            self.model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
        elif algorithm == 'SAC':
            self.model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
        elif algorithm == 'DDPG':
            self.model = DDPG(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
        else:
            raise ValueError('Define proper stable_baseline model')

    def learn(self):
        monitor_env = Monitor(self.env)

        checkpoint_callback = CheckpointCallback(save_freq=self.config['save_freq'], save_path=self.config['save_path'])
        # Separate evaluation env
        eval_callback = EvalCallback(monitor_env, best_model_save_path='best_model',
                                     log_path=self.config['log_path'], eval_freq=self.config['log_interval'])
        # Create the callback list
        callback = CallbackList([checkpoint_callback, eval_callback])

        self.model.learn(total_timesteps=self.config['total_timesteps'], log_interval=self.config['log_interval'],
                         progress_bar=True, tb_log_name=self.config['log_name'], callback=callback)
        self.model.save(f"{self.algorithm}_reacher_env")


    def load_model(self, path):

        self.model.load(path)



