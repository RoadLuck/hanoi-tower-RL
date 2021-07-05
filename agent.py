import numpy as np

class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, epsilon):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon


        self.Q = {}

        #self.init_Q()

    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                print((state,action))
                self.Q[state, action] = 0.0
        print(self.Q)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            action = np.random.choice([i for i in range(self.n_actions)])
            actions = np.array([self.Q[state, a] for a in range(self.n_actions)])
            action = np.argmax(actions)
        return action

    def learn(self, state, action, reward, state_):
        #print("S",state)
        #print("A1",action)
        #print("S'",state_)
        actions = np.array([self.Q[state_, a] for a in range(self.n_actions)])
        a_max = np.argmax(actions)

        self.Q[state, action] += self.lr*(reward + self.gamma*self.Q[state_, a_max] - self.Q[state, action])