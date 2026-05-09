#imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Assuming you have imported TanhGaussianPolicy, Mlp, and ReplayBuffer from core.py
from core import ReplayBuffer, Mlp, TanhGaussianPolicy


class REDQ:
    def __init__(self, obs_dim, act_dim, act_limit, device='mps', hidden_sizes=(256, 256),
                 lr=3e-4, N=5, replay_size=int(1e6), gamma=0.99, blend_weight=0.995,
                 batch_size=256, critics_chosen=2):
        self.device = device
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.N = N
        self.gamma = gamma
        self.blend_weight = blend_weight
        self.batch_size = batch_size
        self.critics_chosen = critics_chosen



        # 1. Initialize the Policy Network (The Actor)
        # We pass the dimensions, hidden sizes, and the action limit (to scale the Tanh output)
        # Then we immediately move the network to the correct hardware device (CPU or GPU)
        self.policy_net = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=act_dim,
            hidden_sizes=hidden_sizes,
            action_limit=act_limit
        ).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) #init optimazer for network

        # 2. Initialize the Critics
        self.q_net_list = []
        self.q_target_net_list = []
        self.q_optimizer_list = []

        for q_i in range(self.N):
            # A. Main Q-Network
            # Input is (state + action), output is 1 (the Q-value)
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(self.device)
            self.q_net_list.append(new_q_net)

            # B. Target Q-Network
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(self.device)
            # Crucial step: strictly copy the initial weights from the main network
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)

            # C. Q-Network Optimizer
            # Each Q-network gets its own Adam optimizer
            self.q_optimizer_list.append(optim.Adam(new_q_net.parameters(), lr=lr))

        # 3. Init replay buffer
        self.replay_buffer = ReplayBuffer(
                 obs_dim=obs_dim,
                 act_dim=act_dim,
                 size=replay_size
             )
        # 4. init temperature alpha parameter
        self.target_entropy = -self.act_dim #target heuristic is the negative of the action dimension
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) #optimize log_alpha so alpha is always positive
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr) #adam optimizer for alpha
        self.alpha = self.log_alpha.cpu().exp().item() #actual scalar to use

    def select_action(self, obs, env=None, deterministic=False, current_steps=0, start_steps=5000):
        """
        given a state observation, returns an action in numpy format.
        """
        with torch.no_grad(): #no training here, we just want to feed a state to the policy network to obtain the action to take

            #the warm-up phase: PICK RANDOM ACTION
            if not deterministic and current_steps < start_steps and env is not None:
                action = env.action_space.sample() #pick random action if we are in the warm-up phase, pure exploration.
                return action

            #using the policy neural network: CHOOSE ACTION GIVEN POLICY
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device) #convert the state (np array) into a PyTorch Tensor

            #pass the input state through the policy network (the actor)
            action_tensor = self.policy_net.forward(
                obs=obs_tensor,
                deterministic=deterministic,
                return_log_prob=False
            )[0]

            action = action_tensor.cpu().numpy().reshape(-1) #convert back to numpy

        return action

    def train(self):
        """

        :return:
        """
        batch = self.replay_buffer.sample_batch(self.batch_size) #sample mini batch of random experiences

        #convert the np arrays into PyTorch tensors and move them to device
        obs = torch.Tensor(batch['obs1']).to(self.device)
        next_obs = torch.Tensor(batch['obs2']).to(self.device)
        acts = torch.Tensor(batch['acts']).to(self.device)

        #turn a 1D array [256] into a 2D column with dims [256, 1]
        rews = torch.Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done = torch.Tensor(batch['done']).unsqueeze(1).to(self.device)

        #Compute REDQ target q-value
        with torch.no_grad():  #target networks weights do not change


            next_acts, _, _, log_prob_next, _, _ = self.policy_net.forward(next_obs) #pick action following the policy nn

            #Feed the current state and action to the main Qnetworks (the critics)
            q_target_preds = []
            for q_target_net in self.q_target_net_list:
                q_target_preds.append(q_target_net(torch.cat([next_obs, next_acts], 1)))

            q_target_preds_cat = torch.cat(q_target_preds, dim=1) #concatenate list of tensors into one large tensor with shape [batch_size, N]

            sample_idxs = np.random.choice(self.N, self.critics_chosen, replace=False) #pick at random critics


            min_q_target, _ = torch.min(q_target_preds_cat[:, sample_idxs], dim=1, keepdim=True) #take the minimum prediction between the subset of critics chosen randomly

            # Bellman Equation + Entropy Bonus
            y_q = rews + self.gamma * (1 - done) * (min_q_target - self.alpha * log_prob_next) #function to minimize (bellman + entropy: SAC equation)

        #update the q main networks (critics)
        q_preds = []
        for q_net in self.q_net_list:
            q_preds.append(q_net(torch.cat([obs, acts], 1)))
        q_preds_cat = torch.cat(q_preds, dim=1)  #shape [batch_size, N]

        y_q_expanded = y_q.expand((-1, self.N))

        q_loss = nn.MSELoss()(q_preds_cat, y_q_expanded) * self.N #calculate MSE for all q networks at once

        #backpropagate and step all q-nets optimizers
        for q_optim in self.q_optimizer_list:
            q_optim.zero_grad()

        q_loss.backward()

        for q_optim in self.q_optimizer_list:
            q_optim.step()

        #update the policy network (actor)
        curr_acts, _, _, log_prob_curr, _, _ = self.policy_net.forward(obs)

        q_curr_preds = []
        for q_net in self.q_net_list: #temporarily freeze critics to evaluate the policy
            q_net.requires_grad_(False)
            q_curr_preds.append(q_net(torch.cat([obs, curr_acts], 1)))

        q_curr_preds_cat = torch.cat(q_curr_preds, dim=1)

        ave_q = torch.mean(q_curr_preds_cat, dim=1, keepdim=True) #REDQ uses the avg of all q-nets for the policy network update
        policy_loss = (self.alpha * log_prob_curr - ave_q).mean() #compute loss

        self.policy_optimizer.zero_grad() #optimizer step
        policy_loss.backward()
        self.policy_optimizer.step()

        #unfreeze the critics for the next loop
        for q_net in self.q_net_list:
            q_net.requires_grad_(True)

        #alpha parameter is learned and modified at each step based on target entropy
        alpha_loss = -(self.log_alpha * (log_prob_curr + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.cpu().exp().item() #alpha scalar

        #target nets update
        with torch.no_grad(): #blending the weights of the main q nets to the target q nets
            for i in range(self.N):
                for param, target_param in zip(self.q_net_list[i].parameters(), self.q_target_net_list[i].parameters()):
                    target_param.data.copy_(self.blend_weight * target_param.data + (1 - self.blend_weight) * param.data)

        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_value': self.alpha
        }

    def save_weights(self, filepath):
        """saves weights of the networks"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'q_net_list': [q_net.state_dict() for q_net in self.q_net_list], #only main qnets, not target ones
            'log_alpha': self.log_alpha
        }, filepath)

    def load_weights(self, filepath):
        """load networks"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.cpu().exp().item()

        for idx, q_net in enumerate(self.q_net_list):
            q_net.load_state_dict(checkpoint['q_net_list'][idx])
            self.q_target_net_list[idx].load_state_dict(q_net.state_dict())

