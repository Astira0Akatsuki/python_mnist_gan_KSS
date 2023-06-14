# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:52:37 2021

@author: USER
"""
# Anaconda Powershell Prompt(anaconda3)나 Anaconda Prompt(anaconda3)에서
# "pip install gym" 설치
# "pip install readchar" 설치

import gym
from gym.envs.registration import register

# # For Linux, Mac
# import sys,tty,termios     
# class _Getch:       
#     def __call__(self):
#             fd = sys.stdin.fileno()
#             old_settings = termios.tcgetattr(fd)
#             try:
#                 tty.setraw(sys.stdin.fileno())
#                 ch = sys.stdin.read(3)
#             finally:
#                 termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#             return ch

# inkey = getkey()

# For Windows          
import readchar

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}

# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name':'4x4','is_slippery':False})


# 여기서부터 gym 코드의 시작이다. 
# env 는 agent가 활동할 수 있는 environment이다
env = gym.make("FrozenLake-v3")
env.render()                    #환경을 화면으로 출력

while True:
    key = readchar.readkey()    #키보드 입력을 받는다

    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]    #에이젼트의 움직임
    state, reward, done, info = env.step(action)    #움직임에 따른 결과값들
    env.render()                #action 후, 화면을 다시 출력
    print("State:", state, "Action", action, "Reward:", reward, "Info:", info)

    if done:                    #도착하면 게임을 끝낸다.
        print("Finished with reward", reward)
        break

# Anaconda Powershell Prompt(anaconda3)나 Anaconda Prompt(anaconda3)에서
# >> ipython rl-l02_01_play_frozenlake_det.py
# >> 화살표를 up, down, right, left로 이동