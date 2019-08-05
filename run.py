import snakeENV2
import agent
import time
import os

env = snakeENV2.ENV()
agent = agent.agent(env.actions)

print("START GAME")
env.startGame()
print("NONE ACTION")
board, rewardLocation, agentLocation, reward, done = env.newAction(None)

print("REMEMBER THE NONE ACTION")
agent.remember(board, agentLocation,rewardLocation, "up", reward, None, None, None, done)
prevBoard, prevRewardLocation, prevAgentLocation = board, rewardLocation, agentLocation


episode = 0
totalReward = 0

while True:
    try:
        episode += 1
        print("EPISODE : ", episode)
        print("REWARD : ", totalReward)

        action = agent.chooseAction(board, agentLocation, rewardLocation)
        
        board, rewardLocation, agentLocation, reward, done = env.newAction(action)
        
        agent.remember(prevBoard, prevAgentLocation, prevRewardLocation, action, reward, board, agentLocation, rewardLocation, done)
        
        prevBoard, prevRewardLocation, prevAgentLocation = board, rewardLocation, agentLocation
        
        if len(agent.memory) >= agent.batchSize:
            agent.learn()
    
    except Exception:
        continue