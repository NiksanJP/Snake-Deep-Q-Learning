import tensorflow as tf
import numpy as np 
import random
import pandas as pd
import time
import sys
import os
import atexit

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
        self.batchSize = 128
        self.learningRate = 0.1
        self.discountRate = 0.1
        self.dropoutRate = 0.1
        
        self.epsilon = 1
        self.minEpsilon = 0.1
        self.epsilon_decay = 0.95
        
        self.count = 0
        self.currentLearnRateCount = 0
        
        self.actions = actions
        
        #Databases
        self.tableAgentReward = self.createTable()
        self.tableBoardReward = self.createTable()
        #Need to change to pandas database
        self.memoryFile = "memory"
        self.memory = pd.DataFrame(columns=['currentBoard', 'action', 'reward', 'rewardLocation', 'newBoard', 'done'])
        self.memoryList = []
        
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
            
        #Exit Handler
        atexit.register(self.exiting)
    
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
    
    def remember(self, currentBoard, action, reward, rewardLocation, newBoard, done):
        currentBoard = self.convertToArray(currentBoard)
        newBoard = self.convertToArray(newBoard)
        
        self.memory = self.memory.append({'currentBoard':currentBoard, 
                            'action':action, 
                            'reward':reward, 
                            'rewardLocation': rewardLocation,
                            'newBoard':newBoard, 
                            'done':done}, ignore_index=True)
    
    def createModel(self):
        #model = tf.keras.models.Sequential([
        #    tf.keras.layers.Dense(32, input_shape=(20,20), activation='relu'),
        #    tf.keras.layers.Flatten(),
        #    tf.keras.layers.Dense(64, activation='relu'),
        #    tf.keras.layers.Dropout(self.dropoutRate),
        #    tf.keras.layers.Dense(128, activation='relu'),
        #    tf.keras.layers.Dropout(self.dropoutRate),
        #    tf.keras.layers.Dense(64, activation='relu'),
        #    tf.keras.layers.Dropout(self.dropoutRate),
        #    tf.keras.layers.Dense(32, activation='relu'),
        #    tf.keras.layers.Dense(4, activation='softmax')
        #])
        
        model = tf.keras.models.Model()
        
        boardInput = tf.keras.layers.Input(shape=(20,20))
        x = tf.keras.layers.Dense(64, activation='relu')(boardInput)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.Model(inputs = boardInput, outputs = x)
        
        rewardInput = tf.keras.layers.Input(shape=(1,2))
        y = tf.keras.layers.Dense(64, activation='relu')(rewardInput)
        y = tf.keras.layers.Flatten()(y)
        y = tf.keras.Model(inputs = rewardInput, outputs = y)
        
        combined = tf.keras.layers.concatenate([x.output,y.output])
        
        hiddenLayer = tf.keras.layers.Dense(64, activation='relu')(combined)
        hiddenLayer = tf.keras.layers.Dropout(self.dropoutRate)(hiddenLayer)
        hiddenLayer = tf.keras.layers.Dense(128, activation='relu')(hiddenLayer)
        hiddenLayer = tf.keras.layers.Dropout(self.dropoutRate)(hiddenLayer)
        hiddenLayer = tf.keras.layers.Dense(64, activation='relu')(hiddenLayer)
        hiddenLayer = tf.keras.layers.Dropout(self.dropoutRate)(hiddenLayer)
        hiddenLayer = tf.keras.layers.Dense(32, activation='relu')(hiddenLayer)
        hiddenLayer = tf.keras.layers.Dense(4, activation='softmax')(hiddenLayer)
        
        model = tf.keras.Model(inputs=[boardInput, rewardInput], outputs = hiddenLayer)

        optimizer = tf.keras.optimizers.Adam()
        model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
        model.summary()
        return model
    
    def convertToArray(self, board):
        for x in range(20):
            for y in range(20):
                if board[x][y] == '_':
                    board[x][y] = 0
                elif board[x][y] == 'S': #Snake
                    board[x][y] = 1
                elif board[x][y] == 'H': #Snake
                    board[x][y] = 2
                elif board[x][y] == 'R': #Reward
                    board[x][y] = 0
        return board
    
    def learn(self, gameLength):
        maxSize = self.memory.shape[0]
        batch = None
        
        print("LEARNING")
        print(self.epsilon)
        
        if maxSize >= gameLength:
            batch = self.memory[-gameLength-100:]
            batch = batch.values.tolist()
            self.currentLearnRateCount += gameLength
            self.batchIndex += self.batchSize

            for currentBoard, action, reward, rewardLocation, newBoard, done in batch:
                
                if action == "right":
                    action = 0
                elif action == "left":
                    action = 1
                elif action == "down":
                    action = 2
                elif action == "up":
                    action = 3
                    
                currentBoard = self.convertToArray(currentBoard)
                currentBoard = np.array(currentBoard)
                currentBoard = np.full((20,20), currentBoard)
                currentBoard = currentBoard[np.newaxis, :, :]
                
                if newBoard is not None:
                    newBoard = self.convertToArray(newBoard)
                else:
                    newBoard = currentBoard
                newBoard = np.array(newBoard)
                newBoard = np.full((20,20), newBoard)
                newBoard = newBoard[np.newaxis, :, :]
                
                
                rewardLocation = np.array([np.array([rewardLocation])])
                finalTarget = self.model.predict([currentBoard,rewardLocation])
                
                #Show fit the action that gives the most reward
                if reward == 5:
                    finalTarget[0][action] += 0.33
                    for i in range(4):
                        if i != action:
                            finalTarget[0][i] -= 0.11
                elif reward == (-15) or reward == (-5): 
                    for i in range(4):
                        if i != action:
                            finalTarget[0][i] += finalTarget[0][action]/3
                    finalTarget[0][action] = 0

                if reward != 0:
                    self.model.fit([currentBoard, rewardLocation], finalTarget, epochs=1, verbose=0)
                
                if self.epsilon > self.minEpsilon and self.currentLearnRateCount >= 100:
                    self.epsilon *= self.epsilon_decay

            self.exiting()
        
    def boardCorrection(self, board):
        board = self.convertToArray(board)
        board = np.array(board)
        board = np.full((20,20), board)
        board = board[np.newaxis, :, :]
        
        return board    
                    
    def chooseAction(self, board, rewardLocation):
        if np.random.rand() <= self.epsilon:
            print("RANDOM ACTION")
            return np.random.choice([0,1,2,3])
        board = self.convertToArray(board)
        board = np.array(board)
        board = np.full((20,20), board)
        board = board[np.newaxis, :, :]
        
        rewardLocation = np.array([np.array([rewardLocation])])
        
        pred = np.argmax(self.model.predict([board, rewardLocation])[0])       
        return pred
    
    def saveWeights(self):
        self.model.save_weights('training/model_weights.h5')
    
    def saveMemory(self):
        self.memory.to_csv('training/memory.csv', index=False)
        print("SAVE ALL SUCCESS")
        
    def exiting(self):
        print("Saving MODEL")    
        self.model.save_weights('training/model_weights.h5')
        self.saveMemory()