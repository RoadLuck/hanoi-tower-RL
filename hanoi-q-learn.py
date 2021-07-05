import gym
import gym_hanoi
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent

if __name__ == '__main__':
    env = gym.make('Hanoi-v0')
    env.num_disk = 3
    
    agent = Agent(lr=0.001, gamma=0.9, epsilon=0.9, n_actions=6, n_states=8)

    scores = []
    win_pct_list = []
    n_games = 1
    #print(env.num_disks)
    observation = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(action)
    print(obs)
    print(reward)

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
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