# FrozenLakeReinforcementLearning

Overview
This project implements a custom version of the Frozen Lake environment and uses Reinforcement Learning (RL) with two neural networks to solve it. Frozen Lake is a classic RL problem where the agent navigates a grid world, avoiding "holes" and reaching the goal.

In this implementation, both the environment and the RL solution are built from scratch without relying on existing libraries like OpenAI Gym.

Features
Custom Environment: A fully custom implementation of the Frozen Lake environment, including state transitions and reward structure.
Deep Q-Network (DQN):
Two neural networks (Main Q-Network and Target Q-Network) are used to predict and update Q-values.
Experience replay is implemented to stabilize training.

ϵ-greedy policy ensures a balance between exploration and exploitation.

Environment
The environment is a 4x4 grid world:

Each state corresponds to a position on the grid.
The agent starts at the top-left corner and must navigate to the goal at the bottom-right corner.
Some grid tiles are "holes" that end the episode with no reward.
Reinforcement Learning
The RL agent uses a Deep Q-Network (DQN):

Main Q-Network:
Predicts Q-values for all possible actions given a state.
Used during training to take actions and compute gradients.
Target Q-Network:
Stabilizes learning by providing fixed Q-value targets for a period of training steps.
Periodically updated to match the weights of the Main Q-Network.
Training
Experience Replay: The agent stores its experiences (state, action, reward, next state, done) in a replay buffer and samples batches for training.
Loss Function: Mean squared error between predicted Q-values and target Q-values.
Optimization: Weights are updated using gradient descent with the Adam optimizer.
