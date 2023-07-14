import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .ReplayBuffer import ReplayBuffer
from .models import Actor, Critic
from .Param_noise import OUNoise
from torch.utils.tensorboard import SummaryWriter

class DDPG_agent:
    def __init__(self, env, config):

        actor_lr = config['actor_lr']
        critic_lr = config['critic_lr']
        loss_type = config['loss_type']
        hidden_size = config['hidden_size']
        replay_buffer_size = config['replay_buffer_size']
        self.batch_size = config['batch_size']
        self.config = config
        self.env = env
        self.device = config['device']

        self.actor_net = Actor(hidden_size, env.observation_space, env.action_space).to(self.device)
        self.actor_target = Actor(hidden_size, env.observation_space, env.action_space).to(self.device)
        self.actor_target.load_state_dict(self.actor_net.state_dict())
        self.actor_net.max_action, self.actor_target.max_action = self.actor_net.max_action, \
        self.actor_target.max_action
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)

        self.critic_net = Critic(hidden_size, env.observation_space, env.action_space).to(self.device)
        self.critic_target = Critic(hidden_size, env.observation_space, env.action_space).to(self.device)
        self.critic_target.load_state_dict(self.critic_net.state_dict())
        self.critic_optim = torch.optim.Adam(self.actor_net.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

       # self.to(self.device)

        # Loss type
        if loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def hard_update(self, target_params, source_params, tau):
        for target_param, param in zip(target_params, source_params):
            target_param.data.copy_(param.data)

    def soft_update(self, target_params, source_params, tau):
        for target_param, param in zip(target_params, source_params):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def predict(self, state):
        state = torch.Tensor(state).to(self.device)
        action = self.actor_target.get_action(state)
        return action

    def update(self):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (done * self.config['gamma'] * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic_net(state, action)

        # Compute critic loss
        self.critic_loss = self.loss(current_Q, target_Q)


        # Optimize the critic
        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        # Compute actor loss
        self.actor_loss = -self.critic_net(state, self.actor_net(state)).mean()


        # Optimize the actor
        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        # update actors
        self.soft_update(self.actor_target.parameters(), self.actor_net.parameters(), self.config['soft_tau'])

        # update critics
        self.soft_update(self.critic_target.parameters(), self.critic_net.parameters(), self.config['soft_tau'])

        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        
        

    def learn(self, test_name='DDPG_test'):
        frame_idx = 0
        episode = 0
        rewards = []
        writer = SummaryWriter()
    
        noise = OUNoise(self.env.action_space, mu=1, max_sigma=7, min_sigma=3)
    
     #   wandb.login(key='0924455b517a41e0e3365f94965a372bea61c868')
      #  wandb.init(name=test_name)
      #  wandb.watch(self.actor_target, log='all', log_freq=100)
       # wandb.watch(self.critic_target, log='all', log_freq=100)
    
        while frame_idx < self.config['max_frames']:
    
            state, done = self.env.reset()
            # state_norm = min_max_scaler(env ,env.prod_map ,state)
            episode_reward = 0
    
            for step in range(self.config['max_steps']):
                if frame_idx > 10000:
                    action = self.predict(state)
                else:
                    action = self.predict(state)
                    action = noise.get_action(action)
    
                next_state, reward, done, done,_ = self.env.step(action)
        
        
                self.replay_buffer.push(state, action, reward, next_state, done)
    
                state_norm = next_state
                episode_reward += reward
                frame_idx += 1
    
                if frame_idx % 100000 == 0 and episode != 0:
                    self.save(frame_idx,directory ='models/')
    
                if done:
                    episode += 1
                    break
    
                if len(self.replay_buffer) > self.config['batch_size']:
                    self.update()
                    writer.add_scalar("Actor_loss", self.actor_loss.cpu().detach().numpy(), frame_idx)
                    writer.add_scalar("Critic loss", self.critic_loss.cpu().detach().numpy(), frame_idx)
                    
                  
    
            print("Total T:{} Episode: {}  Total Reward: \t{:0.2f}".format(frame_idx, episode, float(episode_reward)))
            writer.add_scalar("Episode_reward", episode_reward, frame_idx)
            rewards.append(episode_reward)

    def save(self,frame_idx, directory):
        torch.save(self.actor_net.state_dict(), directory + f'actor_{frame_idx}.pth')
        torch.save(self.critic_net.state_dict(), directory + f'critic_{frame_idx}.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")


    def load(self, directory):
        self.actor_net.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic_net.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
