import os
import time
from collections import deque

import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def train(
    env, agents, brain_name=None, num_agents=1,
    n_episodes=2000, max_steps=1000,
    thr_score=13.0
):
    """Train agent in the environment
    Arguments:
        env {UnityEnvironment} -- Unity Environment
        agent {object} -- Agent to traverse environment
    
    Keyword Arguments:
        brain_name {str} -- brain name for Unity environment (default: {None})
        num_agents {int} -- number of training episodes (default: {1})
        n_episodes {int} -- number of training episodes (default: {2000})
        max_steps {int} -- maximum number of timesteps per episode (default: {1000})
        thr_score {float} -- threshold score for the environment to be solved (default: {30.0})
    """

    # Scores for each episode
    scores = []

    # Last 100 scores
    scores_window = deque(maxlen=100)

    # Average scores & steps after each episode (within window)
    avg_scores = []

    # Best score so far
    best_avg_score = -np.inf

    # Loop over episodes
    for i in range(1, n_episodes+1):

        # Get initial state from environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = np.zeros(num_agents)

        # Reset noise
        agents.reset()

        # Play an episode        
        for _ in range(max_steps):
            action = agents.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agents.step(state, action, rewards, next_state, dones)
            state = next_state
            score += rewards
            if np.any(dones):
                break 

        # Update book-keeping variables
        scores_window.append(np.mean(score))
        scores.append(np.mean(score))
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)
        if avg_score > best_avg_score:
            best_avg_score = avg_score

        # Info for user every 100 episodes
        n_secs = int(time.time() - time_start)
        print(f'Episode {i:6}\t Score: {score:.2f}\t Avg: {avg_score:.2f}\t Best Avg: {best_avg_score:.2f} \t Memory: {len(agent.memory):6}\t Seconds: {n_secs:4}')
        time_start = time.time()

        # Check if done
        if avg_score >= thr_score:
            print(f'\nEnvironment solved in {i:d} episodes!\tAverage Score: {avg_score:.2f}')

            # Save the weights
            torch.save(agents.actor_local.state_dict(), 'logs/weights_actor.pth')
            torch.save(agents.critic_local.state_dict(), 'logs/weights_critic.pth') 

            # Create plot of scores vs. episode
            _, ax = plt.subplots(1, 1, figsize=(7, 5))
            sns.lineplot(range(len(scores)), scores, label='Score', ax=ax)
            sns.lineplot(range(len(avg_scores)), avg_scores, label='Avg Score', ax=ax)
            ax.set_xlabel('Episodes')
            ax.set_xlabel('Score')
            ax.set_title('Environment: {}'.format(env.name))
            ax.legend()
            plt.savefig('./logs/scores_{}.png'.format(env.name))

            break

def test(env, agents, brain_name, checkpoint):
    """Let pre-trained agent play in environment
    
    Arguments:
        env {UnityEnvironment} -- Unity Environment
        agent {object} -- Agent to traverse environment
        brain_name {str} -- brain name for Unity environment (default: {None})
        checkpoint {str} -- filepath to load network weights
    """

    raise NotImplementedError()
