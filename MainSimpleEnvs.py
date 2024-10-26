from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 1.0
NUM_EPISODES = 500
NUM_RUNS = 100

def show(env, current_state, reward=None):
    env.show()
    print(f"Raw state: {current_state}")
    if reward:
        print(f"Reward: {reward}")


def get_action_from_user(valid_actions):
    key = input()
    while key not in valid_actions:
        key = input()
    return valid_actions[key]


def play_simple_env(simple_env):
    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    s = simple_env.reset()
    show(simple_env, s)
    done = False
    while not done:
        print("Action: ", end="")
        action = get_action_from_user(key2action)
        s, r, done = simple_env.step(action)
        show(simple_env, s, r)

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    Q = {}
    returns = []
    action_space = env.action_space

    for _ in range(num_episodes):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(action_space)
            else:
                q_values = [Q.get((state, a), 0.0) for a in action_space]
                max_q = max(q_values)
                max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
                action = np.random.choice(max_actions)
            next_state, reward, done = env.step(action)
            total_reward += reward
            q_values_next = [Q.get((next_state, a), 0.0) for a in action_space]
            max_q_next = max(q_values_next)
            td_target = reward + gamma * max_q_next
            td_error = td_target - Q.get((state, action), 0.0)
            Q[(state, action)] = Q.get((state, action), 0.0) + alpha * td_error
            state = next_state
        returns.append(total_reward)
    return returns

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    Q = {}
    returns = []
    action_space = env.action_space

    for _ in range(num_episodes):
        total_reward = 0
        state = env.reset()
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)
        else:
            q_values = [Q.get((state, a), 0.0) for a in action_space]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
            action = np.random.choice(max_actions)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward
            if np.random.rand() < epsilon:
                next_action = np.random.choice(action_space)
            else:
                q_values_next = [Q.get((next_state, a), 0.0) for a in action_space]
                max_q = max(q_values_next)
                max_actions = [a for a, q in zip(action_space, q_values_next) if q == max_q]
                next_action = np.random.choice(max_actions)
            td_target = reward + gamma * Q.get((next_state, next_action), 0.0)
            td_error = td_target - Q.get((state, action), 0.0)
            Q[(state, action)] = Q.get((state, action), 0.0) + alpha * td_error
            state = next_state
            action = next_action
        returns.append(total_reward)
    return returns

def n_step_sarsa(env, num_episodes, alpha, gamma, epsilon, n):
    Q = {}
    returns = []
    action_space = env.action_space

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)
        else:
            q_values = [Q.get((state, a), 0.0) for a in action_space]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
            action = np.random.choice(max_actions)
        states = [state]
        actions = [action]
        rewards = [0]
        T = float('inf')
        t = 0
        done = False
        while True:
            if t < T:
                next_state, reward, done = env.step(action)
                total_reward += reward
                states.append(next_state)
                rewards.append(reward)
                if done:
                    T = t + 1
                else:
                    if np.random.rand() < epsilon:
                        next_action = np.random.choice(action_space)
                    else:
                        q_values = [Q.get((next_state, a), 0.0) for a in action_space]
                        max_q = max(q_values)
                        max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
                        next_action = np.random.choice(max_actions)
                    actions.append(next_action)
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += gamma ** n * Q.get((states[tau + n], actions[tau + n]), 0.0)
                Q_sa = Q.get((states[tau], actions[tau]), 0.0)
                Q[(states[tau], actions[tau])] = Q_sa + alpha * (G - Q_sa)
            if tau == T - 1:
                break
            t += 1
            if t < T:
                state = states[t]
                action = actions[t]
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    returns_q_learning = np.zeros(NUM_EPISODES)
    returns_sarsa = np.zeros(NUM_EPISODES)
    returns_4step_sarsa = np.zeros(NUM_EPISODES)

    for run in range(NUM_RUNS):
        print(f"Run {run + 1}/{NUM_RUNS}")
        env = CliffEnv()
        returns = q_learning(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        returns_q_learning += np.array(returns)

        env = CliffEnv()
        returns = sarsa(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        returns_sarsa += np.array(returns)

        env = CliffEnv()
        returns = n_step_sarsa(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON, n=4)
        returns_4step_sarsa += np.array(returns)

    returns_q_learning /= NUM_RUNS
    returns_sarsa /= NUM_RUNS
    returns_4step_sarsa /= NUM_RUNS

    plt.plot(returns_q_learning, label='Q-learning')
    plt.plot(returns_sarsa, label='Sarsa')
    plt.plot(returns_4step_sarsa, label='4-step Sarsa')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.ylim(-200, None)
    plt.legend()
    plt.show()

