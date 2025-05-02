# Cliff Walking with TD(0) Policy Evaluation

This project implements **Temporal-Difference (TD(0)) prediction** on the classic **Cliff Walking** environment using the **RL-Glue framework**. It simulates a bootstrapping-based, model-free reinforcement learning agent performing policy evaluation — estimating the value function for fixed policies in an episodic Markov Decision Process (MDP).

---

## Objectives

- Implement a custom Cliff Walking MDP environment
- Build a TD(0) agent for state-value function estimation
- Run policy evaluation experiments for optimal, safe, and stochastic policies
- Visualize value functions, error metrics, and learning dynamics
- 
---

##  Environment Description

The **Cliff Walking** environment is a grid-based world where:

- The agent starts in the bottom-left and must reach the bottom-right goal
- Actions: `Up (0)`, `Left (1)`, `Down (2)`, `Right (3)`
- Rewards:
  - Step: `-1`
  - Falling into the cliff: `-100` and restart from the beginning
- Episode ends when the agent reaches the goal

The grid dimensions are configurable. Default size: `4 x 12`.

---

## Agent: TD(0) Learning

The agent uses the TD(0) update rule for policy evaluation:

```
V[s] += alpha * (reward + gamma * V[next_state] - V[s])

Where:
  alpha is the learning rate
  gamma is the discount factor
  V is the estimated state-value function
  The agent follows a fixed policy provided at initialization.
```

## Experiments

Three types of policies are evaluated:
- Optimal Policy – Minimal steps along the cliff (risky)
- Safe Policy – Avoids cliff by moving along upper boundaries
- Stochastic Policy – A softened version of the optimal policy

## Visualizations include:
- Value heatmaps
- RMSVE (Root Mean Square Value Error) curves
- Action policy arrows
