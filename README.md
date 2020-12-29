# Udacity Reacher Project
Udacity Deep Reinforcement Learning Nanodegree Reacher Project

## Introduction
This is a solution for the second project of the [Udacity deep reinforcement learning course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). It includes a script to train an agent using the TD3 algorithm. The models are trained using the [Stable Baselines3 project](https://stable-baselines3.readthedocs.io/en/master/#).

## Problem description
The agent consists of an arm with two joints and the environment contains a sphere which is rotating around the agent.
The goal is to keep touching the ball as long as possible during an episode of 1000 timesteps.

- Rewards:
  - +0.04 for each timestep the agent touches the sphere
- Input state:
  - 33 continuous variables corresponding to position, rotation, velocity, and angular velocities of the arm
- Actions:
  - 4 continuous variables, corresponding to torque applicable to two joints with values in [-1.0, 1.0]
- Goal:
  - Get an average score of at least +30 over 100 consecutive episodes
- Environment:
  -  The environment that is used is a single agent provided by Udacity. 

## Solution
The problem is solved with TD3 using the [stable baselines framework](https://stable-baselines3.readthedocs.io/en/master/).
 
## Setup project
Unfortunately, the Unity ML environment used by Udacity for this project is a very early version that is a few years old - [v0.4](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0). This makes it extremely difficult to set things up, particularly in a Windows environment. Please see the separate guide I have provided on how to do this: [setup project](Setup.md).

## Training an agent
I have implemented an experiment based framework that allows for exploration of different hyperparameters when training a model. The parameters that you can specify are these:

- **learning-rate** the learning rate to use for training the model. Default is 0.0003.
- **batch-size** the size of batches that are sampled and used to train the model. Default is 100.
- **buffer-size** the size of the replay buffer. Default is 100,000.
- **total-timesteps** the total number of timesteps to train for. Default is 100,000.
- **seed** random seed to use. Default is -1 which means generate a random value.
- **environment-port** this is the port number used for communication with the Unity environment. If you want to have more than one agent running at the same time you would specify a different port for each of them. Default is 5005.
- **policy-layers** the hidden layers to use in the neural network. Specified as a comma separated list. Default is "400,300".
- **algorithm** the name of the algorithm to use to train the agent. At the moment only td3 is supported and this is the default value.
- **executable-path** the path to the executable to run. Please see the notes about [setup project](Setup.md) to help with this.
- **experiments-root** the root folder that experiment output will be written to.
- **experiment-name** the name of the experiment. A subfolder will be created to the 'experiments-root' folder with this name and all output from this experiment will be written to it.
- **reward-threshold** the reward value that the solution must attain over 100 consecutive episodes. Default is 30.0.
- **gamma** the gamma value used for greedy strategy. Default is 0.99.


Here is an example command line:

```python train_agent.py --experiment-name td3-lr_0_0005-128_128_128_ts_600K --learning-rate 0.0005 --total-timesteps 600000 --policy-layers "128,128,128"```
