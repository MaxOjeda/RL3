from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
from MainSimpleEnvs import play_simple_env
import numpy as np
import matplotlib.pyplot as plt
import random

ALPHA = 0.1
EPSILON = 0.1
GAMMA = 0.99
NUM_EPISODES = 500
NUM_RUNS = 100

def play_room_env():
    n_episodes = 10
    for _ in range(n_episodes):
        env = RoomEnv()
        play_simple_env(env)

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    Q = {}  # Initialize Q-values to 1.0
    episode_lengths = []
    action_space = env.action_space

    for episode in range(num_episodes):
        state = env.reset()
        total_steps = 0
        done = False

        while not done:
            s, g = state
            if random.random() < epsilon:
                action = random.choice(action_space)
            else:
                q_values = [Q.get((s, g, a), 1.0) for a in action_space]
                max_q = max(q_values)
                max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
                action = random.choice(max_actions)

            next_state, reward, done = env.step(action)
            s_prime, _ = next_state
            total_steps += 1

            # Q-learning update
            q_sa = Q.get((s, g, action), 1.0)
            if done:
                target = reward
            else:
                q_values_next = [Q.get((s_prime, g, a), 1.0) for a in action_space]
                max_q_next = max(q_values_next)
                target = reward + gamma * max_q_next
            Q[(s, g, action)] = q_sa + alpha * (target - q_sa)

            state = next_state

        episode_lengths.append(total_steps)

    return episode_lengths


def sarsa(env, num_episodes, alpha, gamma, epsilon):
    Q = {}
    episode_lengths = []
    action_space = env.action_space

    for episode in range(num_episodes):
        state = env.reset()
        s, g = state
        total_steps = 0

        # Choose action
        if random.random() < epsilon:
            action = random.choice(action_space)
        else:
            q_values = [Q.get((s, g, a), 1.0) for a in action_space]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
            action = random.choice(max_actions)

        done = False
        while not done:
            next_state, reward, done = env.step(action)
            s_prime, _ = next_state
            total_steps += 1

            if not done:
                if random.random() < epsilon:
                    next_action = random.choice(action_space)
                else:
                    q_values_next = [Q.get((s_prime, g, a), 1.0) for a in action_space]
                    max_q = max(q_values_next)
                    max_actions = [a for a, q in zip(action_space, q_values_next) if q == max_q]
                    next_action = random.choice(max_actions)
            else:
                next_action = None

            # Sarsa update
            q_sa = Q.get((s, g, action), 1.0)
            if done:
                target = reward
            else:
                q_sprime_aprime = Q.get((s_prime, g, next_action), 1.0)
                target = reward + gamma * q_sprime_aprime
            Q[(s, g, action)] = q_sa + alpha * (target - q_sa)

            s, action = s_prime, next_action

        episode_lengths.append(total_steps)

    return episode_lengths


def n_step_sarsa(env, num_episodes, alpha, gamma, epsilon, n):
    Q = {}
    episode_lengths = []
    action_space = env.action_space

    for episode in range(num_episodes):
        state = env.reset()
        s, g = state
        total_steps = 0
        T = float('inf')
        t = 0

        # Initialize lists
        states = [s]
        actions = []
        rewards = [0]

        # Choose initial action
        if random.random() < epsilon:
            action = random.choice(action_space)
        else:
            q_values = [Q.get((s, g, a), 1.0) for a in action_space]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
            action = random.choice(max_actions)
        actions.append(action)
        done = False

        while True:
            if t < T:
                next_state, reward, done = env.step(actions[t])
                s_prime, _ = next_state
                states.append(s_prime)
                rewards.append(reward)
                total_steps += 1

                if done:
                    T = t + 1
                else:
                    if random.random() < epsilon:
                        next_action = random.choice(action_space)
                    else:
                        q_values_next = [Q.get((s_prime, g, a), 1.0) for a in action_space]
                        max_q = max(q_values_next)
                        max_actions = [a for a, q in zip(action_space, q_values_next) if q == max_q]
                        next_action = random.choice(max_actions)
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    q_s_ap = Q.get((states[tau + n], g, actions[tau + n]), 1.0)
                    G += gamma ** n * q_s_ap
                q_sa = Q.get((states[tau], g, actions[tau]), 1.0)
                Q[(states[tau], g, actions[tau])] = q_sa + alpha * (G - q_sa)

            if tau == T - 1:
                break
            t += 1

        episode_lengths.append(total_steps)

    return episode_lengths


def multi_goal_q_learning(env, num_episodes, alpha, gamma, epsilon):
    Q = {}
    episode_lengths = []
    action_space = env.action_space
    goals = env.goals

    for episode in range(num_episodes):
        state = env.reset()
        total_steps = 0
        done = False

        while not done:
            s, g = state
            if random.random() < epsilon:
                action = random.choice(action_space)
            else:
                q_values = [Q.get((s, g, a), 1.0) for a in action_space]
                max_q = max(q_values)
                max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
                action = random.choice(max_actions)

            next_state, reward, done = env.step(action)
            s_prime, _ = next_state
            total_steps += 1

            # Update for all goals
            for goal in goals:
                q_sa = Q.get((s, goal, action), 1.0)
                if s_prime != goal:
                    q_values_next = [Q.get((s_prime, goal, a), 1.0) for a in action_space]
                    max_q_next = max(q_values_next)
                    target = gamma * max_q_next
                else:
                    target = 1.0
                Q[(s, goal, action)] = q_sa + alpha * (target - q_sa)

            state = next_state

        episode_lengths.append(total_steps)

    return episode_lengths


def multi_goal_sarsa(env, num_episodes, alpha, gamma, epsilon):
    Q = {}
    episode_lengths = []
    action_space = env.action_space
    goals = env.goals

    for episode in range(num_episodes):
        state = env.reset()
        s, g = state
        total_steps = 0

        # Choose action
        if random.random() < epsilon:
            action = random.choice(action_space)
        else:
            q_values = [Q.get((s, g, a), 1.0) for a in action_space]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(action_space, q_values) if q == max_q]
            action = random.choice(max_actions)

        done = False
        while not done:
            next_state, reward, done = env.step(action)
            s_prime, _ = next_state
            total_steps += 1

            if not done:
                if random.random() < epsilon:
                    next_action = random.choice(action_space)
                else:
                    q_values_next = [Q.get((s_prime, g, a), 1.0) for a in action_space]
                    max_q = max(q_values_next)
                    max_actions = [a for a, q in zip(action_space, q_values_next) if q == max_q]
                    next_action = random.choice(max_actions)
            else:
                next_action = None

            # Update for all goals
            for goal in goals:
                q_sa = Q.get((s, goal, action), 1.0)
                if s_prime != goal:
                    if done:
                        q_sprime_aprime = 0.0
                    else:
                        q_sprime_aprime = Q.get((s_prime, goal, next_action), 1.0)
                    target = gamma * q_sprime_aprime
                else:
                    target = 1.0
                Q[(s, goal, action)] = q_sa + alpha * (target - q_sa)

            s, action = s_prime, next_action

        episode_lengths.append(total_steps)

    return episode_lengths

if __name__ == '__main__':
    algorithms = ['Q-learning', 'Sarsa', '8-step Sarsa', 'Multi-goal Q-learning', 'Multi-goal Sarsa']
    total_episode_lengths = {alg: np.zeros(NUM_EPISODES) for alg in algorithms}

    for run in range(NUM_RUNS):
        print(f"Run {run + 1}/{NUM_RUNS}")

        # Q-learning
        env = RoomEnv()
        episode_lengths = q_learning(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        total_episode_lengths['Q-learning'] += np.array(episode_lengths)

        # Sarsa
        env = RoomEnv()
        episode_lengths = sarsa(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        total_episode_lengths['Sarsa'] += np.array(episode_lengths)

        # 8-step Sarsa
        env = RoomEnv()
        episode_lengths = n_step_sarsa(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON, n=8)
        total_episode_lengths['8-step Sarsa'] += np.array(episode_lengths)

        # Multi-goal Q-learning
        env = RoomEnv()
        episode_lengths = multi_goal_q_learning(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        total_episode_lengths['Multi-goal Q-learning'] += np.array(episode_lengths)

        # Multi-goal Sarsa
        env = RoomEnv()
        episode_lengths = multi_goal_sarsa(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        total_episode_lengths['Multi-goal Sarsa'] += np.array(episode_lengths)

    # Compute average episode lengths
    avg_episode_lengths = {alg: total_episode_lengths[alg] / NUM_RUNS for alg in algorithms}

    # Plotting
    plt.figure()
    for alg in algorithms:
        plt.plot(avg_episode_lengths[alg], label=alg)
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Length')
    plt.legend()
    plt.show()
