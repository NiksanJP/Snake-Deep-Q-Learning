import tensorflow as tf
import numpy as np 
import random
import pandas as pd
import time
import sys
import os

# Disclaimer from the Author himself
# "What is going on in this code. Even me the author fails to understand after a few months"
# Good luck mate

class agent:
    def __init__(self, actions):
        """ 
        NAIVE CONCEPT
           Change in weights = lr X [ ( reward + discount Rate X prediction of next value ) ) - predicted value of current value ] X Gradient of current Q Value
           Q target = reward of taking the action at the state + Discounted max Q Value among all possible actions from next state
           Q target => Q(s,a) = reward(s,a) + g X maxQ(s2, a)
    
        DEEP Q Learning
            maybe but s -> state and a -> action
           Q target Q(s,a) = r(s,a) + g X Q(s2, argmax Q(s2, a))
        """
        
        #GAME PROPERTIES
        self.batchSize = 16
        self.learningRate = 0.05
        self.discountRate = 0.9
        self.dropoutRate = 0.05
        self.epsilon = 0.9
        self.minEpsilon = 0.3
        self.epsilon_decay = 0.9
        self.count = 0
        self.currentLearnRateCount = 0
        
        self.actions = actions
        
        #Databases
        self.tableAgentReward = self.createTable()
        self.tableBoardReward = self.createTable()
        self.memoryFile = "memory"
        self.memory = []
        
        #Deep Learning 
        self.batchSize = 2
        self.batchIndex = 2
        self.model = self.createModel()
        
        checkPointPath = "training/checkpoint"
        self.checkPointDIR = os.path.dirname(checkPointPath)
        
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkPointDIR,
                                                             save_weights_only=True,
                                                             verbose=0)
        
        try:
            self.model.load_weights('training/model_weights.h5')
            print("################################################")
            print("#############Training Data Loaded###############")
            print("################################################")
            self.model.load_weights(checkPointPath)
            print("################################################")
            print("#############Training Data Loaded###############")
            print("################################################")
        except Exception as ex:
            print(ex)
            pass
        #LOAD MEMORY
        try:
            data = pd.read_csv("memory.csv")
            self.memory = data.values.tolist()
        except Exception as ex:
            print(ex)
            pass
    
    def createTable(self):
        return pd.DataFrame(columns = self.actions, dtype = np.float64)
    
    def checkStateExistsBoard(self, state):
        if state not in self.tableBoardReward.index:
            self.tableBoardReward = self.tableBoardReward.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.tableBoardReward.columns,
                    name=state,
                )
            )
            
    def checkStateExistAgent(self, state):
        if state not in self.tableAgentReward.index:
            self.tableAgentReward = self.tableAgentReward.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.tableAgentReward.columns,
                    name=state,
                )
            )
    
    def remember(self, currentBoard, currentAgent, currentRewardLocation, action, reward, newBoard, newAgent, newRewardLocation, done):
        self.memory.append([    currentBoard,
                                currentAgent,
                                currentRewardLocation,
                                action,
                                reward,
                                newBoard,
                                newAgent,
                                newRewardLocation,
                                done])
        
        #Save everything in the Q Table
        self.checkStateExistAgent(str( [newBoard, newRewardLocation] ) )
        self.checkStateExistsBoard( str( [newAgent, newRewardLocation] ))
    
    def createModel(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, input_shape=(20,20), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(self.dropoutRate),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(self.dropoutRate),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(self.dropoutRate),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(self.dropoutRate),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(self.dropoutRate),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation="softmax")
        ])

        optimizer = tf.keras.optimizers.Adadelta()
        model.compile( loss = 'mse', optimizer = optimizer)
        model.summary()
        return model
    
    def convertToArray(self, board):
        for x in range(20):
            for y in range(20):
                if board[x][y] == '_':
                    board[x][y] = 0
                elif board[x][y] == 'S': #Snake
                    board[x][y] = 1
                elif board[x][y] == 'R': #Reward
                    board[x][y] = 2
        return board
    
    def learn(self):
        model = self.model
        maxSize = len(self.memory)
        batch = None
        
        print("LEARNING")
        
        if len(self.memory) >= self.batchSize:
            batch = self.memory[-self.currentLearnRateCount:]
            self.currentLearnRateCount += 5000
            self.batchIndex += self.batchSize
            
            for currentBoard, currentAgent, currentRewardLocation, action, reward, newBoard, newAgent, newRewardLocation, done in batch:
                
                currentBoard = self.convertToArray(currentBoard)
                currentBoard = np.array(currentBoard)
                currentBoard = np.full((20,20), currentBoard)
                currentBoard = currentBoard[np.newaxis, :, :]
                
                currentRewardLocation = np.array(currentRewardLocation)
                currentRewardLocation = np.full((1,2), currentRewardLocation)
                currentRewardLocation = currentRewardLocation[np.newaxis, :, :]

                if newBoard is not None:
                    newBoard = self.convertToArray(newBoard)
                else:
                    newBoard = currentBoard
                newBoard = np.array(newBoard)
                newBoard = np.full((20,20), newBoard)
                newBoard = newBoard[np.newaxis, :, :]
                
                newRewardLocation = np.array(newRewardLocation)
                newRewardLocation = np.full((1,2), newRewardLocation)
                newRewardLocation = newRewardLocation[np.newaxis, :, :]

                target = reward
                
                x = newBoard, newRewardLocation
                y = currentBoard, currentRewardLocation
                
                if not done:
                    target = ( reward + self.discountRate * np.amax(model.predict(x)) )

                finalTarget = model.predict(y)
                
                if action == "right":
                    action = 0
                elif action == "left":
                    action = 1
                elif action == "down":
                    action = 2
                elif action == "up":
                    action = 3
                
                finalTarget[0][action] = target
                
                model.fit([currentBoard, currentRewardLocation], [finalTarget], epochs=1, verbose=0)
                
                if self.epsilon > self.minEpsilon:
                    self.epsilon *= self.epsilon_decay
        
        print("Saving MODEL")    
        model.save_weights('training/model_weights.h5')
        
        print("GIVING BACK TO MODEL")
        self.model = model
                    
    def chooseAction(self, board, agent, rewardLocation):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.actions)
        board = self.convertToArray(board)
        board = np.array(board)
        board = np.full((20,20), board)
        board = board[np.newaxis, :, :]
                
        rewardLocation = np.array(rewardLocation)
        rewardLocation = np.full((1,2), rewardLocation)
        rewardLocation = rewardLocation[np.newaxis, :, :]
        pred = np.argmax(self.model.predict([board, rewardLocation])[0])
        return pred
    
    def saveWeights(self):
        self.model.save_weights('training/model_weights.h5')
    
    def saveMemory(self):
        df = pd.DataFrame(self.memory, colummns =["column"])
        df.to_csv('memory.csv', index=False)
        self.tableAgentReward.to_csv('tableAgentReward.csv', index=False)
        self.tableBoardReward.to_csv('tableBoardReward.csv', index=False)
        print("SAVE ALL SUCCESS")