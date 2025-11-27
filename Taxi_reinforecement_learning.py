# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 23:37:01 2021

@author: gaurang
"""

# Load OpenAI Gym and other necessary packages
# import gym
import gymnasium as gym
import numpy as np
import time

# Environment for training (no rendering)
env = gym.make("Taxi-v3")

# Training parameters for Q-learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode

# Parameters for epsilon-greedy strategy
epsilon = 1.0                 # Exploration rate
epsilon_decay_rate = 0.001    # Rate of decay for epsilon
min_epsilon = 0.01            # Minimum exploration rate

# Q tables for rewards
q_table = np.zeros((env.observation_space.n, env.action_space.n))

for episode in range(num_of_episodes):
    state, info = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            # Explore: select a random action
            action = env.action_space.sample()
        else:
            # Exploit: select the best action
            action = np.argmax(q_table[state,:])

        new_state, reward, done, truncated, info = env.step(action)

        # Update Q-table using the Bellman equation
        q_table[state, action] = q_table[state, action] * (1 - alpha) + alpha * (reward + gamma * np.max(q_table[new_state, :]))

        state = new_state
    
    # Decay epsilon to reduce exploration over time
    epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
#taxi_row, taxi_col, passenger_location, destination = env.encode(state);
#print(taxi_row)
# Testing

reward_list = []
action_list = []

# Create a new environment for testing with rendering enabled
test_env = gym.make("Taxi-v3", render_mode="human")

for _ in range(20):
    state, info = test_env.reset()
    total_reward = 0.0
    action_count = 0
    done = False
    truncated = False

    for t in range(50):  # default 50
        time.sleep(0.1)
        action = np.argmax(q_table[state,:])
        state, reward, done, truncated, info = test_env.step(action)
        total_reward += float(reward)
        action_count += 1
        if done or truncated:
            print(f"Total reward: {total_reward}")
            reward_list.append(total_reward)
            action_list.append(action_count)
            print(f"Total Actions {action_count}")
            break

test_env.close()

action_list_np = np.array(action_list)
reward_list_np = np.array(reward_list)

print("------------------------------------------------")
print("Averages are : - ")
print(f"All Actions :- {action_list_np}")
print(f"All Rewards :- {reward_list_np}")
print(f"Average Actions {np.mean(action_list_np)}")
print(f"Average Rewards {np.mean(reward_list_np)}")

# Add this line to pause the script before closing
input("Press Enter to exit...")