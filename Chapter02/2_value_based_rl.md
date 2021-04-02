---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    executable: /usr/bin/env python
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: tfrl-cookbook
    language: python
    name: tfrl-cookbook
---

Value-based reinforcement learning


Chapter 2, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

```python
import numpy as np
```

```python
from envs.maze import MazeEnv
from value_function_utils import visualize_maze_values
```

```python
discount = 0.9
iters = 1000
```

```python
env = MazeEnv()
state_dim = env.distinct_states
state_values = np.zeros(state_dim)
q_values = np.zeros((state_dim, env.action_space.n))  # Action values
policy = np.zeros(state_dim)
```

```python
def calculate_values(state, action):
    """Evaluate Value function for given state and action

    Args:
        state (int): Valid (discrete) state in discrete `env.observation_space`
        action (int): Valid (discrete) action in `env.action_space`

    Returns:
        v_sum: value for given state, action
    """
    v_sum = 0
    transitions = []
    slip_action = env.slip_action_map[action]
    env.set_state(state)
    slip_next_state, slip_reward, _ = env.step(slip_action, slip=False)
    transitions.append((slip_reward, slip_next_state, env.slip))
    env.set_state(state)
    next_state, reward, _ = env.step(action, slip=False)
    transitions.append((reward, next_state, 1 - env.slip))
    for reward, next_state, pi in transitions:
        v_sum += pi * (reward + discount * state_values[next_state])
    return v_sum
```

```python
# Value Iteration
for i in range(iters):
    v_s = np.zeros(state_dim)
    for state in range(state_dim):
        if env.index_to_coordinate_map[int(state / 8)] == env.goal_pos:
            continue
        v_max = float("-inf")
        for action in range(env.action_space.n):
            v_sum = calculate_values(state, action)
            v_max = max(v_max, v_sum)
        v_s[state] = v_max
    state_values = np.copy(v_s)
```

```python
for state in range(state_dim):
    for action in range(env.action_space.n):
        q_values[state, action] = calculate_values(state, action)
```

```python
for state in range(state_dim):
    policy[state] = np.argmax(q_values[state, :])
```

```python
np.save("results/q_values", q_values)
np.save("results/optimal_policy", policy)
# print(q_values)
print("Action mapping:[0 - UP; 1 - DOWN; 2 - LEFT; 3 - RIGHT")
print("Optimal actions:")
print(policy)
visualize_maze_values(q_values, env)
```
