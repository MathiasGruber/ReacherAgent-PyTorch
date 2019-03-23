# Reacher Game Report
As so many others I've based my implementation on the deep deterministic gradient method (DDPG), [see paper](https://arxiv.org/abs/1509.02971v5).

Settings used to solve the 1-agent and 20-agent environment. Values are inspired by [Zulkhuu repo](https://github.com/Zulkhuu/reinforcement-learning/blob/master/Reacher/docs/Report.md), especially in terms of number of hidden nodes in network layers, where the original paper values of [400, 300] were too much to give meaningful results.
```
BATCH_SIZE = 128        # Batch Size
BUFFER_SIZE = int(1e5)  # Memory capacity
GAMMA = 0.99            # Discount factor
LR_ACTOR = 1e-4         # Actor lerning rate
LR_CRITIC = 1e-4        # Critic learning rate
TAU = 1e-3              # Soft update of target networks
WEIGHT_DECAY = 0        # L2 weight decay for Critic
NOISE_SIGMA = 0.2       # sigma for Ornstein-Uhlenbeck noise

# Actor
dense_layers=[256, 128]

# Critic
dense_layers=[256, 128]
```