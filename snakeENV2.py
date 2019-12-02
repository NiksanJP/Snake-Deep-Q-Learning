import numpy
import random
import numpy as np
import time
import sys
import cv2

class ENV:
    def __init__(self):       
        #AGENT
        self.agent = [[10,10],[10,11],[10,12] ]
        self.headAgent = self.agent[0]
        self.run = False
        self.actions = ["right", "left", "down", "up"]
        self.primMove = True
        self.reward = 0
        
        #Move
        self.primMove = "up"
        self.directions = {
            "left" : [-1, 0],
            "right": [1, 0],
            "up": [0, 1],
            "down": [0, -1],
        }
        
        #Reward
        self.rewardLocationX = random.randint(1,18)
        self.rewardLocationY = random.randint(1,18)
        self.prevRewardX = self.rewardLocationX
        self.prevRewardy = self.rewardLocationY
        
        #BOARD
        self.boardSizeX = 500
        self.boardSizeY = 500
        
        self.boardDimX = 20
        self.boardDimY = 20
        
        self.textRep = True
        self.board = [
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ]
        

    def getScreen(self):
        self.bg = cv2.imread('bg_White.png')
        self.bg = cv2.resize(self.bg, (500,500))
        return self.bg
        
    def startGame(self):
        if self.run == False:
            self.run = True
        
        self.colorRed = (255,0,0)
        self.colorGreen = (0,255,0)
        self.colorBlue = (0,0,255)
        self.colorDarkBlue = (0,0,128)
        self.colorWhite = (255,255,255)
        self.colorYellow = (255,204,0)
        self.colorOrange = (255,100,71)
        self.colorBlack = (0,0,0)
        self.colorPink = (255,200,200)
        self.colorBgColor_white = (255,255,255)
        self.colorBgColor_black = (0,0,0)
    
    def rewardChange(self, r):
        if r == 0:
            r = -10
        elif r < 0: 
            r -= 10
        elif r > 0:
            r *= 10
        
        return r
            
    def newAction(self, action):
        done = False
        
        if action == None:
            action = self.primMove
        
        if action == 0:
            action = "right"
        elif action == 1:
            action = "left"
        elif action == 2:
            action = "down"
        elif action == 3:
            action = "up"
        
        try:
            direction = self.directions[action]
        except:
            direction = self.directions[self.primMove]
            
        a = self.agent[0][0] + direction[0]
        b = self.agent[0][1] + direction[1]    
        
        #backing
        if [a,b] == self.agent[0]:
            print("BACKING")
            self.reward = -200
            
            self.reward = self.rewardChange(self.reward)
            
            return self.board, [self.rewardLocationX, self.rewardLocationY], self.agent, self.reward, done
        
        #Reward
        if [a,b] == [self.rewardLocationX, self.rewardLocationY]:
            self.reward = 50
            newBody = self.agent[-1:][0]
            print(newBody)
            self.agent.insert(0, [newBody[0], newBody[1]])
            self.newReward()
        
        #Touch own body
        if [a,b] in self.agent:
            print("TOUCH OWN BODY")
            self.reward = -100
            done = True
            print("REWARD : ", self.reward)
            return self.board, [self.rewardLocationX, self.rewardLocationY], self.agent, self.reward, done
            
            self.reset()
            self.newReward()
            done = True
            
            self.updateBoard()  
            self.printBoard()
            
            self.reward = self.rewardChange(self.reward)
            
            return self.board, [self.rewardLocationX, self.rewardLocationY], self.agent, self.reward, done
            
        #Touch a Wall 
        if a == 20 or a == -1 or b == -1 or b == 20:
            print("OUTSIDE")
            self.reward = -20
            self.reset()
            self.newReward()
            done = True
            
            self.updateBoard()  
            self.printBoard()
            
            self.reward = self.rewardChange(self.reward)
            
            return self.board, [self.rewardLocationX, self.rewardLocationY], self.agent, self.reward, done
        
        if a != 19 or a != 0 or b != 0 or b != 19:
            self.agent.insert(0, [a,b])
            self.agent = self.agent[:-1]

        self.reward = self.rewardChange(self.reward)
        print("REWARD : ", self.reward)

        self.updateBoard()  
        self.printBoard()
        return self.board, [self.rewardLocationX, self.rewardLocationY], self.agent, self.reward, done
    
    def reset(self):
        self.agent = [[10,10],[10,11],[10,12]]
        self.headAgent = self.agent[0]
        self.run = True
        self.reward = 0
        
        self.board = [
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ]
        
        self.updateBoard()
    
    def newReward(self):
        self.rewardLocationX = random.randint(1,19)
        self.rewardLocationY = random.randint(1,19)  
            
        while [self.rewardLocationX, self.rewardLocationY] in self.agent:
            self.rewardLocationX = random.randint(1,19)
            self.rewardLocationY = random.randint(1,19)    
                       
    def updateBoard(self):
        self.board = [
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ]

        for x,y in self.agent:
            self.board[x][y] = "S"
        
        self.board[self.rewardLocationX][self.rewardLocationY] = "R"
                    
    def printBoard(self):
        self.bg = self.getScreen()
        
        for i in range(len(self.agent)):
            x1 = (self.agent[i] [1] * 25) + 1
            y1 = (self.agent[i] [0] * 25) + 1
            
            x2 = x1 + 16
            y2 = y1 + 16

            if i == 0:
                self.bg = cv2.rectangle(self.bg, (x1,y1), (x2,y2), self.colorOrange, -1)
            else:
                self.bg = cv2.rectangle(self.bg, (x1,y1), (x2,y2), self.colorBlack, -1)
        
        #REWARD
        y1 = (self.rewardLocationX * 25) + 1
        x1 = (self.rewardLocationY * 25) + 1

        x2 = x1 + 16
        y2 = y1 + 16
        
        self.bg = cv2.rectangle(self.bg, (x1,y1), (x2,y2), (0,0,255), -1)
        
        cv2.imwrite('pic.png', self.bg)
        img = cv2.imread('pic.png')  
        cv2.imshow('pic', img)
        cv2.waitKey(1)
                    