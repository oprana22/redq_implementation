import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

# Note: the contents of this python file are extracted directly from the source code. Since the paper includes little
# details about the networks and buffer themselves, for the sake of reproducibility and common sense, I decided to use the given classes.

#math stability constraints for the policy network
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6

def weights_init_(m):
    """
    weight initialization helper function
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1) #uses Xavier uniform initialization to linear layers to prevent gradient problems
        torch.nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    """
    numpy FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        """stores a transition at the current pointer location"""
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        #move the pointer to store in the next location, looping back if full
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        """
        samples a random mini batch of experiences
        """
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)


class Mlp(nn.Module):
    """
    MLP used for the critic nets
    """

    def __init__(self, input_size, output_size, hidden_sizes, hidden_activation=F.relu):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation

        self.hidden_layers = nn.ModuleList() #use ModuleList so PyTorch tracks the parameters correctly
        in_size = input_size

        #initialize each hidden layer dynamically
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        #initialize final output layer
        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init_)

    def forward(self, input):
        #forward pass
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output


class TanhNormal(Distribution):
    """
    represents distribution of X where:
        X ~ tanh(Z)
        Z ~ N(mean, std)
    required for the SAC reparameterization trick
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        calculates the log probability of a value
        """

        if pre_tanh_value is None:  #check if the pre-activation value was not provided
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)) / 2  #calculate the inverse tanh to get the original value
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon)  #return the probability using the change of variables formula

    def sample(self, return_pretanh_value=False):
        """
        draws a sample from the distribution
        """
        z = self.normal.sample().detach()  #draw a random sample from the normal distribution and detach it from the gradient graph
        if return_pretanh_value:
            return torch.tanh(z), z  #return both the squashed action and the raw sample
        else:
            return torch.tanh(z)  #return only the squashed action

    def rsample(self, return_pretanh_value=False):
        """
        sampling with the reparameterization trick
        """
        #reparametrization trick: sampling standard normal noise and scaling it by the standard deviation then adding the mean, to make it differentiable
        z = (self.normal_mean + self.normal_std * Normal(torch.zeros(self.normal_mean.size()),
                                                         torch.ones(self.normal_std.size())).sample())
        if return_pretanh_value:
            return torch.tanh(z), z #return both the squashed action and the raw sample
        else:
            return torch.tanh(z) #return only the squashed action


class TanhGaussianPolicy(Mlp): #policy neural net (actor) inherits from the mlp class  of the qvalue nets (critics)
    """
    a gaussian policy network with tanh to enforce action limits
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes, hidden_activation=F.relu, action_limit=1.0):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )

        last_hidden_size = obs_dim #set last hidden size to the observation dimension
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1] #update to the size of the final hidden layer if there are hidden layers

        #parallel layer to output the log standard deviation
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim) #create a parallel linear layer to predict the log standard deviation
        self.action_limit = action_limit #store the environment maximum action value for scaling
        self.apply(weights_init_) #apply the xavier uniform weight initialization to all layers

    def forward(self, obs, deterministic=False, return_log_prob=True):
        """
        define the forward pass to map states to actions
        """
        h = obs #start the forward pass with the observation tensor
        for fc_layer in self.hidden_layers: #iterate through each hidden layer
            h = self.hidden_activation(fc_layer(h)) #apply the linear transformation and the activation function

        mean = self.last_fc_layer(h) #calculate the mean of the action distribution using the final base layer
        log_std = self.last_fc_log_std(h) #calculate the log standard deviation using the parallel layer

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX) #restrict the log standard deviation to prevent math explosions
        std = torch.exp(log_std) #convert log standard deviation to standard deviation

        normal = Normal(mean, std) #create a normal distribution using the calculated mean and std

        if deterministic: #if "evaluation mode"
            pre_tanh_value = mean #use the raw mean without adding noise
            action = torch.tanh(mean) #squash the mean directly to get the deterministic action
        else: #if "training mode"
            pre_tanh_value = normal.rsample() #sample with the reparameterization trick to allow gradient flow
            action = torch.tanh(pre_tanh_value) #squash the sampled value to bound it between -1 and 1

        if return_log_prob: #if the entropy calculation is needed
            log_prob = normal.log_prob(pre_tanh_value) #calculate the probability of the raw sample
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON) #apply the jacobian correction for the tanh squashing function
            log_prob = log_prob.sum(1, keepdim=True) #sum the probabilities across the action dimensions
        else:
            log_prob = None

        return action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value #return the scaled action and distribution parameters