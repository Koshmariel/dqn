#%% INITIALIZATION: libraries, parameters, network...


from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves 

import numpy as np
import gym                                # To train our network
env = gym.make('MountainCar-v0')          # Choose game (any in the gym should work)

import random     # For sampling batches from the observations


model = Sequential()


model.add(Dense(20, input_shape=(env.observation_space.shape[0], ), activation="relu", kernel_initializer="uniform"))

model.add(Dense(18, kernel_initializer="uniform", activation="relu"))

model.add(Dense(10, kernel_initializer="uniform", activation="relu"))

model.add(Dense(env.action_space.n, activation="linear", kernel_initializer="uniform"))    # Same number of outputs as possible actions

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Parameters
D = deque()                               # Register where the states will be stored

epsilon = 1.                              # Probability of doing a random move
epsilon_dec=1                             # Epsilon decrement factor
gamma = 0.99                              # Discounted future reward. How much we care about steps further in time
mb_size = 100                             # Learning minibatch size


#%%# Learning while playing
n_games = 10000
ddqn_scores = []
eps_history = []


tot_reward = 0.0
finished = 0
reports = []
reports_finished = []

for i in range(n_games):
    if i % 100 == 0:
        print('Game', i)
    done = False
    score = 0
    state = env.reset()

    
    step = 0
    while not done:
        
        step = step + 1
        #env.render()
        
        obs = np.expand_dims(state, axis=0)
       
        rand_num = np.random.random()
        
        if rand_num < epsilon:
            action = np.random.choice(env.action_space.n)     #random action
            action_rnd = True
        else:
            Q = model.predict(obs)
            action = np.argmax(Q)
            action_rnd = False
        epsilon = epsilon*(epsilon_dec-0.01*finished)
        
        
        
        state_new, reward, done, info = env.step(action)
        D.append((state, action, reward, state_new, int(done)))         # 'Remember' action and consequence
        score += reward
        
        report = f'Game {i} Step {step} Epsilon {epsilon:.2f} Action {action} random={action_rnd} reward {reward} score {score} finished {finished}'
        reports.append(report)
        
        if (done == True) and (step !=200):
            finished = finished + 1
            reports_finished.append(report)
            #epsilon_dec = epsilon_dec * 0.999
            

#                             returns list of tuples
        minibatch = np.array(random.sample(D, min(mb_size,len(D))))
        states = minibatch[:,0]

        states=np.stack(states) #transforms array of arrays into 2d array
        
        actions = minibatch[:,1]
        actions = np.array(actions, dtype=np.int32)
        rewards = minibatch[:,2]
        states_new = minibatch[:,3]
        states_new=np.stack(states_new) #transforms array of arrays into 2d array
        dones = minibatch[:,4]
        Q_eval = model.predict(states)
        
        Q_next = model.predict(states_new)
        
        Q_target = Q_eval.copy()
        batch_index = np.arange(min(mb_size,len(D)), dtype=np.int32)
                                                                                #invert dones
        Q_target[batch_index, actions] = rewards + gamma*np.max(Q_next, axis=1)*(1-dones)
        
        _ = model.fit(states, Q_target, verbose=0)
        
        
        state = state_new
        
        
