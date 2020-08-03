import gym
import slimevolleygym

env = gym.make("SlimeVolley-v0")

def run(policy):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
      action = policy(obs)
      obs, reward, done, info = env.step(action)
      total_reward += reward
      env.render()

    print("score:", total_reward)

def compare(policy1, policy2):
    obs1 = env.reset()
    obs2 = obs1  # both sides always see the same initial observation.

    done = False
    total_reward = 0

    while not done:
        action1 = policy1(obs1)
        action2 = policy2(obs2)

        obs1, reward, done, info = env.step(action1, action2)  # extra argument
        obs2 = info['otherObs']

        total_reward += reward
        env.render()

    print("policy1's score:", total_reward)
    print("policy2's score:", -total_reward)