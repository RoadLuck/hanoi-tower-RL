import gym
import gym_hanoi
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent

if __name__ == '__main__':
    env = gym.make('Hanoi-v0')
    env.set_env_parameters(num_disks=2)

    actions = env.action_space.n
    states = len(env.observation_space)*3

    
    agent = Agent(lr=0.85, gamma=0.95, n_actions=actions, n_states=states, eps_start=0.9, eps_end=0.1, eps_dec=0.999)

    scores = []
    win_pct_list = []
    n_games = 10

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            print(action)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_)
            score += reward
            observation = observation_
            scores.append(score)
    if i % 100 == 0:
        win_pct = np.mean(scores[-100:])
        win_pct_list.append(win_pct)
    if i % 1000 == 0:
            print('episode ', i, 'win pct %.2f' % win_pct,
                'epsilon %.2f' % agent.epsilon)
    plt.plot(win_pct_list)
    plt.show()
