# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the Q-Learning control algorithm in the Gymnasium FrozenLake-v1 environment. The agent must learn the optimal action-value function through repeated interaction with the environment and determine the best action to take in each state. The objective is to reach the goal state while avoiding the hole states and to evaluate the learning performance using the learned policy, state-value function, average reward, and learning curve.

## Software Requirements
```
	Python 3.x
	Gymnasium
	NumPy
	Matplotlib
	Jupyter Notebook / Google Colab / Python IDE
  Operating System
```


## Environment Description
FrozenLake-v1 is a 4×4 grid-world environment with 16 states. The agent starts at S (Start) and must reach G (Goal) while avoiding H (Holes). The agent can move Left, Down, Right, or Up. Reaching the goal gives a reward of 1, while falling into a hole gives 0 reward and ends the episode.


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
```
1.Create the FrozenLake-v1 environment and initialize the Q-table with zeros.
2.Set the learning parameters: learning rate (α), discount factor (γ), and ε.
3.Reset the environment and observe the current state.
4.Select an action using the epsilon-greedy method.
5.Perform the action and observe the reward and next state.

6.Update the Q-value using the Q-Learning formula:

$$ Q(S,A) \leftarrow Q(S,A)+\alpha[R+\gamma\max Q(S',a)-Q(S,A)] $$
7.Repeat until the episode ends.
8.Gradually decrease ε and repeat for multiple episodes.
9.Select the action with the highest Q-value to obtain the learned policy.
10.Evaluate the agent using the average reward and learning curve.
```


## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        # Choose action using epsilon-greedy
        action = choose_action(state)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Q-Learning update
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] = Q[state, action] + alpha * (
            target - Q[state, action]
        )

        state = next_state
        total_reward += reward

    # Store episode reward
    episode_rewards.append(total_reward)

    # Reduce epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)








```
---

## Output

Final Q-table:

<img width="263" height="350" alt="image" src="https://github.com/user-attachments/assets/28a4e861-e95f-42aa-8d45-81c162320af5" />





Estimated State-Value Function:

<img width="290" height="115" alt="image" src="https://github.com/user-attachments/assets/e738e130-20ba-432a-8669-ba8dfc77dba8" />





Learned Policy:

<img width="187" height="113" alt="image" src="https://github.com/user-attachments/assets/092a9128-8c76-43a1-a2b9-b1a4ccc795e1" />



Average reward over last 1000 episodes: 

<img width="415" height="22" alt="image" src="https://github.com/user-attachments/assets/c0695196-66d2-44a6-9a5b-162c96d3c3ec" />


---

## Result

```text

The Q-Learning algorithm was successfully implemented using the
FrozenLake-v1 environment. The agent learned the Q-values for
different state-action pairs and generated a learned policy for
reaching the goal while avoiding the holes. The learning performance
was evaluated using the average reward and learning curve.

```

---

## Inference

```text

The experiment shows that Q-Learning enables an agent to learn the
best actions through repeated interaction with the environment.
The Q-table improves with training, allowing the agent to select
suitable actions and achieve better rewards. The epsilon-greedy
strategy helps balance exploration and exploitation during learning.

```

---

