import argparse
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import torch
import copy

def build_net(layer_shape, hidden_activation, output_activation):
    layers = []
    for j in range(len(layer_shape)-1):
        act = hidden_activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.Tanh, output_activation=nn.Tanh):
        super(Actor, self).__init__()

        self.layer_sizes = [state_dim] + list(hid_shape)
        # Building hidden layers dynamically
        self.net = build_net(self.layer_sizes, hidden_activation, output_activation)

        self.mu_layer = nn.Linear(hid_shape[-1], action_dim)
        self.log_std_layer = nn.Linear(hid_shape[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=False):
        '''Network with Enforcing Action Bounds'''
        x = state
        for layer in self.net:
            x = layer(x)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        a = torch.tanh(u)

        if with_logprob:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a

class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2

#reward engineering for better training
def Reward_adapter(r, EnvIdex):
    return r

def Action_adapter(a,max_action):
    #from [-1,1] to [-max,max]
    return  a*max_action

def Action_adapter_reverse(act,max_action):
    #from [-max,max] to [-1,1]
    return act/max_action

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class SAC_countinuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = Actor(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
        print(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
        print(self.q_critic)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def train(self, writer, total_steps):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()
        if self.write: writer.add_scalar('q_loss', q_loss, global_step=total_steps)

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters(): params.requires_grad = False

        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        if self.write: writer.add_scalar('a_loss', a_loss, global_step=total_steps)
        for params in self.q_critic.parameters(): params.requires_grad = True

        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        if self.write: writer.add_scalar('alpha', self.alpha, global_step=total_steps)
        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self,EnvName, timestep):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))
        torch.save(self.m_var.state_dict(), "./model/{}_m_var{}.pth".format(EnvName,timestep))

    def load(self,EnvName, timestep):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))
        self.m_var.load_state_dict(torch.load("./model/{}_m_var{}.pth".format(EnvName, timestep)))

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
        self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) 
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size 
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

import gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
def main():
    # Initialize the environment
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  
    max_action = float(env.action_space.high[0]) 

    # Define hyperparameters
    hid_shape = (256, 256)  # Hidden layer sizes
    a_lr = 3e-4            # Actor learning rate
    c_lr = 3e-4            # Critic learning rate
    batch_size = 256       # Batch size for training
    alpha = 0.2            # Initial alpha value
    adaptive_alpha = True  # Use adaptive alpha
    gamma = 0.99           # Discount factor
    write = True          # tensorboard 
    dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "./runs"  
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create SAC agent
    agent = SAC_countinuous(
        state_dim=state_dim,
        action_dim=action_dim,
        hid_shape=hid_shape,
        a_lr=a_lr,
        c_lr=c_lr,
        batch_size=batch_size,
        alpha=alpha,
        adaptive_alpha=adaptive_alpha,
        gamma=gamma,
        write=write,
        dvc=dvc
    )

    # Training loop
    num_episodes = 1000  # Limited episodes for testing
    total_steps = 0
    with tqdm(range(num_episodes), desc='Training Progress') as tbar:
        for episode in tbar:
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                # Select and scale action
                action = agent.select_action(state, deterministic=False)  
                action_scaled = Action_adapter(action, max_action)    
                
                # Step environment
                next_state, reward, done, _, _ = env.step(action_scaled)
                
                # Store experience
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                # Train if buffer has enough samples
                if agent.replay_buffer.size > batch_size:
                    agent.train(writer=writer, total_steps=total_steps)
            tbar.set_postfix(episode_reward=episode_reward)
            if write:
                writer.add_scalar('Reward/episode_reward', episode_reward, episode)

if __name__ == "__main__":
    main()
