from collections import defaultdict
import numpy as np
import gym

def generate_episode(env, policy ,Q):
    episode = []
    state = env.reset()
    while True:
        action = policy(state, env.action_space.n, env, Q)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

env = gym.make('CliffWalking-v0')
Q = defaultdict(lambda: np.zeros(env.action_space.n))
# print(generate_episode(env, sarsa, Q))