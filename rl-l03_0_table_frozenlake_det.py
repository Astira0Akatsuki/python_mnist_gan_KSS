# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:52:37 2021

@author: USER
"""
# Anaconda Powershell Prompt(anaconda3)나 Anaconda Prompt(anaconda3)에서
# "pip install gym" 설치
# "pip install readchar" 설치

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)                   # random 선택


# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name':'4x4','is_slippery':False})


# 여기서부터 gym 코드의 시작이다. 
# env는 agent가 활동할 수 있는 environment 생성이다
env = gym.make("FrozenLake-v3")

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])     # 16 x 4 array
# Set learning parameters
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False
    
    # The Q-Table learning algorithm
    while not done:                                 # done이 되지 않는 한 무한 루프
        #에이젼트의 움직임, 랜덤하게 argmax가 최대인 곳으로 이동
        action = rargmax(Q[state, :])               
        
        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)   #움직임에 따른 결과값들
        
        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + np.max(Q[new_state, : ])
        
        rAll += reward
        state = new_state
        
    rList.append(rAll)
    
# Result reporting
print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()