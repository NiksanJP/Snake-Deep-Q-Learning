import snakeENV2
import agent
import time
import os
import threading
import atexit
import subprocess

env = snakeENV2.ENV()
agent = agent.agent(env.actions)

print("START GAME")
env.startGame()
print("NONE ACTION")
board, rewardLocation, agentLocation, reward, done = env.newAction(None)

print("REMEMBER THE NONE ACTION")
prevBoard, prevRewardLocation, prevAgentLocation = board, rewardLocation, agentLocation

episode = 0
totalReward = 0
game = 0
gameLength = 0

while True:
    episode += 1
    print("EPISODE : ", episode)
    print("GAME : ", game)

    action = agent.chooseAction(board, rewardLocation)
    
    board, rewardLocation, agentLocation, reward, done = env.newAction(action)
    
    agent.remember(prevBoard, action, reward, rewardLocation, board, done)
    
    prevBoard, prevRewardLocation, prevAgentLocation = board, rewardLocation, agentLocation
    
    if done:
        game += 1
    gameLength += 1
    if done and game % 1 == 0:
        agent.learn(gameLength)  
        gameLength = 0
    
    os.system('cls' if os.name == 'nt' else 'clear')