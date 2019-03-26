# ReacherAgent-PyTorch
This model is developed as a solution to Project 2 of Udacity Deep Reinforcement Learning Nanodegree

<img src="logs/trained_agent.gif" width="50%" /><br />
<sup>Image from <a href="https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control">official repo</a></sup>

# Installation
Install the package requirements for this repository
```
pip install -r requirements.txt
```

# Reacher environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Two separate versions of the Unity environment are provided:

* The first version contains a single agent. The task is episodic, the agent must get an average score of +30 over 100 consecutive episodes for it to be considered solved.
* The second version contains 20 identical agents, each with its own copy of the environment. The task is episodic, and an average score for the 20 agents is calculated after each episode. The environment is considered solved when the average (over 100 episodes) of those average scores is at least +30.

The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

# Getting the environment
Download your specific environment and unpack it into the `./env_unity/` folder in this repo:

- **_Version 1: One (1) Agent_**
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **_Version 2: Twenty (20) Agents_**
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

# Training the agent
For training the agent on the single-agent version of the environment, the model can be run by using one of the following (only tested on windows!):
```
python main.py --environment env_unity/Reacher_singleAgent/Reacher.exe --memory per
python main.py --environment env_unity/Reacher_singleAgent/Reacher.exe
```

For training the agent on the 20-agent version of the environment, the following can be used (opnly tested on windows!):
```
python main.py --environment env_unity/Reacher_multipleAgents/Reacher.exe --memory per
python main.py --environment env_unity/Reacher_multipleAgents/Reacher.exe
```

# Testing the agent
Once the agent has been trained, it can be run as follows:
```
python main.py --environment env_unity/Reacher_multipleAgents/Reacher.exe --test --checkpoint_actor logs/weights_actor_multipleAgents_per.pth --checkpoint_critic logs/weights_critic_multipleAgents_per.pth
```