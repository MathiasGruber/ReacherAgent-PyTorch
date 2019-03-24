# Reacher Game Report
As so many others I've based my implementation on the deep deterministic gradient method (DDPG), [see paper](https://arxiv.org/abs/1509.02971v5).

## Learning Algorithm

## Hyperparameters
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

## Results / Plots of Rewards
### Normal replay buffer
<p float="left">
  <img src="logs/scores_singleAgent_replay.png" width="48%" />
  <img src="logs/scores_multipleAgents_replay.png" width="48%" />
</p>

### Prioritized experience replay
<p float="left">
  <img src="logs/scores_singleAgent_per.png" width="48%" />
  <img src="logs/scores_multipleAgents_per.png" width="48%" />
</p>

## Future Work
The main addition to off-the-shelf DDPG in this repository is prioritized experience replay. For future improvements of the DDPG agent. 
- [ ] I'd look into further tuning hyperparameters
- [ ] I'd try reducing the Ornstein-Uhlenbeck noise as more episodes are played.
- [ ] Due to the instability of DDPG, I might look into its extensions such as [Self-Adaptive Double Bootstrapped DDPG (SOUP)](https://www.ijcai.org/proceedings/2018/0444.pdf) that advertise better training stability.