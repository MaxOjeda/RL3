from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv

from MainSimpleEnvs import show, get_action_from_user
import numpy as np
import matplotlib.pyplot as plt
import random

ALPHA = 0.1
EPSILON = 0.1
GAMMA = 0.95
NUM_EPISODES = 50000
NUM_RUNS = 30
EPISODE_LOG_INTERVAL = 100  # Average over every 100 episodes

def q_learning_centralized(env, num_episodes, alpha, gamma, epsilon):
    Q = {}
    action_space = env.action_space
    total_episode_lengths = np.zeros(num_episodes // EPISODE_LOG_INTERVAL)

    for episode in range(num_episodes):
        state = env.reset()
        total_steps = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(action_space)
            else:
                q_values = [Q.get((state, action), 1.0) for action in action_space]
                max_q = max(q_values)
                max_actions = [action for action, q in zip(action_space, q_values) if q == max_q]
                action = random.choice(max_actions)

            next_state, reward, done = env.step(action)
            total_steps += 1

            # Q-learning update
            q_sa = Q.get((state, action), 1.0)
            if done:
                target = reward
            else:
                q_values_next = [Q.get((next_state, a), 1.0) for a in action_space]
                max_q_next = max(q_values_next)
                target = reward + gamma * max_q_next
            Q[(state, action)] = q_sa + alpha * (target - q_sa)

            state = next_state

        # Record episode length
        idx = episode // EPISODE_LOG_INTERVAL
        total_episode_lengths[idx] += total_steps

    avg_episode_lengths = total_episode_lengths / EPISODE_LOG_INTERVAL
    return avg_episode_lengths

def q_learning_decentralized_cooperative(env, num_episodes, alpha, gamma, epsilon):
    Q1 = {}
    Q2 = {}
    action_space = env.single_agent_action_space
    total_episode_lengths = np.zeros(num_episodes // EPISODE_LOG_INTERVAL)

    for episode in range(num_episodes):
        state = env.reset()
        total_steps = 0
        done = False

        while not done:
            # Agent 1 action
            if random.random() < epsilon:
                action1 = random.choice(action_space)
            else:
                q_values1 = [Q1.get((state, action1), 1.0) for action1 in action_space]
                max_q1 = max(q_values1)
                max_actions1 = [action1 for action1, q in zip(action_space, q_values1) if q == max_q1]
                action1 = random.choice(max_actions1)

            # Agent 2 action
            if random.random() < epsilon:
                action2 = random.choice(action_space)
            else:
                q_values2 = [Q2.get((state, action2), 1.0) for action2 in action_space]
                max_q2 = max(q_values2)
                max_actions2 = [action2 for action2, q in zip(action_space, q_values2) if q == max_q2]
                action2 = random.choice(max_actions2)

            action = (action1, action2)
            next_state, rewards, done = env.step(action)
            reward = rewards[0]  # Both agents receive the same reward
            total_steps += 1

            # Update Q-values
            q_sa1 = Q1.get((state, action1), 1.0)
            q_sa2 = Q2.get((state, action2), 1.0)
            if done:
                target = reward
            else:
                q_values_next1 = [Q1.get((next_state, a), 1.0) for a in action_space]
                q_values_next2 = [Q2.get((next_state, a), 1.0) for a in action_space]
                max_q_next1 = max(q_values_next1)
                max_q_next2 = max(q_values_next2)
                target = reward + gamma * max_q_next1  # Same for both agents

            Q1[(state, action1)] = q_sa1 + alpha * (target - q_sa1)
            Q2[(state, action2)] = q_sa2 + alpha * (target - q_sa2)

            state = next_state

        # Record episode length
        idx = episode // EPISODE_LOG_INTERVAL
        total_episode_lengths[idx] += total_steps

    avg_episode_lengths = total_episode_lengths / EPISODE_LOG_INTERVAL
    return avg_episode_lengths

def q_learning_decentralized_competitive(env, num_episodes, alpha, gamma, epsilon):
    Q_hunter1 = {}
    Q_hunter2 = {}
    Q_prey = {}
    action_space = env.single_agent_action_space
    total_episode_lengths = np.zeros(num_episodes // EPISODE_LOG_INTERVAL)

    for episode in range(num_episodes):
        state = env.reset()
        total_steps = 0
        done = False

        while not done:
            # Hunter 1 action
            if random.random() < epsilon:
                action1 = random.choice(action_space)
            else:
                q_values1 = [Q_hunter1.get((state, action1), 1.0) for action1 in action_space]
                max_q1 = max(q_values1)
                max_actions1 = [action1 for action1, q in zip(action_space, q_values1) if q == max_q1]
                action1 = random.choice(max_actions1)

            # Hunter 2 action
            if random.random() < epsilon:
                action2 = random.choice(action_space)
            else:
                q_values2 = [Q_hunter2.get((state, action2), 1.0) for action2 in action_space]
                max_q2 = max(q_values2)
                max_actions2 = [action2 for action2, q in zip(action_space, q_values2) if q == max_q2]
                action2 = random.choice(max_actions2)

            # Prey action
            if random.random() < epsilon:
                action_prey = random.choice(action_space)
            else:
                q_values_prey = [Q_prey.get((state, action_prey), 1.0) for action_prey in action_space]
                max_q_prey = max(q_values_prey)
                max_actions_prey = [action_prey for action_prey, q in zip(action_space, q_values_prey) if q == max_q_prey]
                action_prey = random.choice(max_actions_prey)

            action = (action1, action2, action_prey)
            next_state, rewards, done = env.step(action)
            reward_hunter = rewards[0]  # Both hunters receive the same reward
            reward_prey = rewards[2]
            total_steps += 1

            # Update hunters
            q_sa1 = Q_hunter1.get((state, action1), 1.0)
            q_sa2 = Q_hunter2.get((state, action2), 1.0)
            if done:
                target_hunter = reward_hunter
            else:
                q_values_next1 = [Q_hunter1.get((next_state, a), 1.0) for a in action_space]
                q_values_next2 = [Q_hunter2.get((next_state, a), 1.0) for a in action_space]
                max_q_next1 = max(q_values_next1)
                max_q_next2 = max(q_values_next2)
                target_hunter = reward_hunter + gamma * max_q_next1  # Same for both hunters

            Q_hunter1[(state, action1)] = q_sa1 + alpha * (target_hunter - q_sa1)
            Q_hunter2[(state, action2)] = q_sa2 + alpha * (target_hunter - q_sa2)

            # Update prey
            q_sa_prey = Q_prey.get((state, action_prey), 1.0)
            if done:
                target_prey = reward_prey
            else:
                q_values_next_prey = [Q_prey.get((next_state, a), 1.0) for a in action_space]
                max_q_next_prey = max(q_values_next_prey)
                target_prey = reward_prey + gamma * max_q_next_prey

            Q_prey[(state, action_prey)] = q_sa_prey + alpha * (target_prey - q_sa_prey)

            state = next_state

        # Record episode length
        idx = episode // EPISODE_LOG_INTERVAL
        total_episode_lengths[idx] += total_steps

    avg_episode_lengths = total_episode_lengths / EPISODE_LOG_INTERVAL
    return avg_episode_lengths

def play_hunter_env():
    hunter_env = HunterAndPreyEnv()

    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down", '': "None"}
    num_of_agents = hunter_env.num_of_agents
    s = hunter_env.reset()
    show(hunter_env, s)
    done = False
    while not done:
        print("Hunter A: ", end="")
        hunter1 = get_action_from_user(key2action)
        print("Hunter B: ", end="")
        hunter2 = get_action_from_user(key2action)
        action = hunter1, hunter2
        if num_of_agents == 3:
            print("Prey: ", end="")
            prey = get_action_from_user(key2action)
            action = hunter1, hunter2, prey
        s, r, done = hunter_env.step(action)
        show(hunter_env, s, r)


if __name__ == "__main__":
    runs_centralized = []
    runs_decentralized_coop = []
    runs_decentralized_comp = []

    for run in range(NUM_RUNS):
        print(f"Run {run + 1}/{NUM_RUNS}")

        # Centralized Cooperative
        env = CentralizedHunterEnv()
        avg_lengths_centralized = q_learning_centralized(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        runs_centralized.append(avg_lengths_centralized)

        # Decentralized Cooperative
        env = HunterEnv()
        avg_lengths_decentralized_coop = q_learning_decentralized_cooperative(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        runs_decentralized_coop.append(avg_lengths_decentralized_coop)

        # Decentralized Competitive
        env = HunterAndPreyEnv()
        avg_lengths_decentralized_comp = q_learning_decentralized_competitive(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
        runs_decentralized_comp.append(avg_lengths_decentralized_comp)

    # Average over runs
    avg_centralized = np.mean(runs_centralized, axis=0)
    avg_decentralized_coop = np.mean(runs_decentralized_coop, axis=0)
    avg_decentralized_comp = np.mean(runs_decentralized_comp, axis=0)

    # Plotting
    episodes = np.arange(0, NUM_EPISODES, EPISODE_LOG_INTERVAL)
    plt.plot(episodes, avg_centralized, label='Centralized Cooperative')
    plt.plot(episodes, avg_decentralized_coop, label='Decentralized Cooperative')
    plt.plot(episodes, avg_decentralized_comp, label='Decentralized Competitive')
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Length')
    plt.legend()
    plt.show()