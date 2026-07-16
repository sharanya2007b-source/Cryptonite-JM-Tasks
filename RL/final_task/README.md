# Deep RL and Policy Optimization: CartPole-v1

This repository contains my implementation of three popular deep reinforcement learning algorithms—**DQN**, **A2C**, and **PPO**—using **PyTorch**. All models were trained and evaluated on the **CartPole-v1** environment from Gymnasium.

The main goal of this project was to understand how these algorithms work under the hood rather than simply achieving high scores. Instead of relying on existing RL libraries, I implemented each algorithm from scratch to get a better understanding of concepts like experience replay, target networks, actor-critic learning, entropy regularization, and PPO's clipped objective.

---

## Repository Structure

- **`common.py`**  
  Contains utilities shared across all algorithms, including the replay buffer, vectorized environment setup, logging functions, and other helper utilities.

- **`dqn.py`**  
  Implementation of the Deep Q-Network (DQN) algorithm using experience replay and a target network for more stable learning.

- **`a2c.py`**  
  Implementation of the Advantage Actor-Critic (A2C) algorithm with synchronous vectorized environments for faster and more stable training.

- **`ppo.py`**  
  Implementation of Proximal Policy Optimization (PPO) using the clipped surrogate objective and multiple optimization epochs.

- **`results/`** *(generated after training)*  
  Stores CSV logs containing training statistics and evaluation results.

- **`*.png`**  
  Learning curves and comparison plots generated from the training logs.

---

## Requirements

Install the required Python packages using:

```bash
pip install torch numpy gymnasium matplotlib pandas
```

---

## What I Learned

Working on this project helped me understand the practical challenges involved in training reinforcement learning agents. Some of the key ideas I explored include:

- Why experience replay improves sample efficiency in DQN.
- How target networks help stabilize value learning.
- The role of advantage estimation in actor-critic methods.
- Why adding entropy encourages better exploration.
- How PPO's clipping mechanism prevents unstable policy updates.

Implementing each algorithm from scratch gave me a much clearer understanding of the trade-offs between value-based and policy-based reinforcement learning methods.

---

## Results

Each algorithm was trained on the CartPole-v1 environment, and training metrics were logged throughout the learning process. The repository also includes plots comparing learning performance and illustrating how each algorithm converges over time.

---

## Notes

This project focuses on learning and implementation rather than optimization for benchmark performance. The code is written to be easy to follow, making it useful for anyone interested in understanding the fundamentals of deep reinforcement learning.
