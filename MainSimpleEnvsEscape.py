from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

GAMMA = 1.0
ALPHA = 0.5
EPSILON = 0.1
NUM_EPISODES = 20
NUM_RUNS = 5
PLANNING_STEPS = [0, 1, 10, 100, 1000, 10000]


def dyna_q(env, num_episodes, alpha, gamma, epsilon, planning_steps):
    Q = {}
    model = {}
    returns = np.zeros(num_episodes)
    action_space = env.action_space

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(action_space)
            else:
                q_values = [Q.get((state, a), 0.0) for a in action_space]
                max_q = max(q_values)
                actions_with_max_q = [a for a, q in zip(action_space, q_values) if q == max_q]
                action = random.choice(actions_with_max_q)

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Update Q-value
            q_sa = Q.get((state, action), 0.0)
            max_q_next = max([Q.get((next_state, a), 0.0) for a in action_space])
            Q[(state, action)] = q_sa + alpha * (reward + gamma * max_q_next - q_sa)

            # Update model
            model[(state, action)] = (reward, next_state)

            # Planning
            for _ in range(planning_steps):
                s, a = random.choice(list(model.keys()))
                r, s_prime = model[(s, a)]
                max_q_s_prime = max([Q.get((s_prime, a_prime), 0.0) for a_prime in action_space])
                Q_sa = Q.get((s, a), 0.0)
                Q[(s, a)] = Q_sa + alpha * (r + gamma * max_q_s_prime - Q_sa)

            state = next_state

        returns[episode] += total_reward

    return returns


def rmax_algorithm(env, num_episodes, gamma, m, rmax):
    Q = {}
    N = {}
    model = {}
    returns = np.zeros(num_episodes)
    action_space = env.action_space

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action with highest estimated Q-value
            q_values = [Q.get((state, a), rmax) for a in action_space]
            max_q = max(q_values)
            actions_with_max_q = [a for a, q in zip(action_space, q_values) if q == max_q]
            action = random.choice(actions_with_max_q)

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Update counts and model
            N[(state, action)] = N.get((state, action), 0) + 1
            model[(state, action)] = (reward, next_state)

            # If we've observed this state-action enough times, update Q-values
            if N[(state, action)] == m:
                Q = rmax_value_iteration(Q, model, action_space, gamma, rmax)

            state = next_state

        returns[episode] += total_reward

    return returns


def rmax_value_iteration(Q, model, action_space, gamma, rmax, theta=0.01):
    # Get all possible states
    states = set(s for s, a in model.keys()) | set(s_prime for r, s_prime in model.values())
    V = {s: rmax for s in states}

    while True:
        delta = 0
        for s in states:
            v = V[s]
            q_values = []
            for a in action_space:
                if (s, a) in model:
                    r, s_prime = model[(s, a)]
                    q = r + gamma * V.get(s_prime, rmax)
                else:
                    q = rmax
                q_values.append(q)
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    # Update Q-values
    for s in states:
        for a in action_space:
            if (s, a) in model:
                r, s_prime = model[(s, a)]
                Q[(s, a)] = r + gamma * V.get(s_prime, rmax)
            else:
                Q[(s, a)] = rmax

    return Q

if __name__ == "__main__":
    results = {}

    # Dyna-Q experiments with different planning steps
    print(f"Dyna")
    for planning_steps in tqdm(PLANNING_STEPS):
        all_returns = np.zeros(NUM_EPISODES)
        for run in range(NUM_RUNS):
            env = EscapeRoomEnv()
            returns = dyna_q(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON, planning_steps)
            all_returns += returns
        average_returns = all_returns / NUM_RUNS
        results[('Dyna', planning_steps)] = average_returns

    # RMax experiments with different planning steps
    print("RMax")
    m = 2  # Threshold for number of visits
    rmax = 0  # Set to the maximum possible reward
    for planning_steps in tqdm(PLANNING_STEPS):
        all_returns = np.zeros(NUM_EPISODES)
        for run in range(NUM_RUNS):
            env = EscapeRoomEnv()
            returns = rmax_algorithm(env, NUM_EPISODES, GAMMA, m, rmax, planning_steps)
            all_returns += returns
        average_returns = all_returns / NUM_RUNS
        results[('RMax', planning_steps)] = average_returns

    # Print results
    print("Average Return per Episode:")
    print("Method\tPlanning Steps\tAverage Return")
    for key in results:
        method, planning_steps = key
        avg_return = np.mean(results[key])
        print(f"{method}\t{planning_steps}\t\t{avg_return:.2f}")


