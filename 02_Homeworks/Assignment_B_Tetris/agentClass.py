import numpy as np
import random
import math
import h5py
import matplotlib.pyplot as plt
import scienceplots
import datetime
import json
import torch


plt.style.use('science')

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions and Q-table and storage for the rewards
        # This function should not return a value, store Q table etc as attributes of self
        self.stateTracker = []
        self.reward_tots = []
        self.totalrewards = 0
        self.totalRewardTracker = []
        self.gameLength = 0
        self.gameLengthTracker = []
        self.stateTracker = []
        self.actionTracker = []


        self.actionIsTaken = False

        ## number of states is the 4x4 grid + the four different tiles
        ## binary we can represent this as 2^(16+2) = 2^18
        self.numberOfStates = 2 ** (gameboard.N_row * gameboard.N_col + 2)

        ## number of actions
        ## Translation      (None, Left, LeftLeft, Right)
        ## Rotation         (None, 90, 180, 270)
        numberOfPossibleTranslations = 4
        numberOfPossibleRotations = 4
        self.numberOfActions = numberOfPossibleTranslations * numberOfPossibleRotations
    
        ## initialize the Q-table
        self.Q = np.zeros((self.numberOfStates, self.numberOfActions))
        
        # self.fn_read_state()

        ## initialize the Q-table with random values
        # self.Q = np.random.randint(0, 5, self.Q.shape)

        print("Q-table and agent initialized")
        print("Shape of Q-table: ", self.Q.shape)


        self.bestReward = -150
        self.bestQ = np.zeros((self.numberOfStates, self.numberOfActions))
        
        self.last_actionID = 0



        self.sessionIndex = self.readLogJSON("index") + 1

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.episode_count' the total number of episodes in the training

    def fn_load_strategy(self,strategy_file):
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)
        self.Q = np.load(strategy_file)

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as an integer entry in the Q-table
        # This function should not return a value, store the state as an attribute of self

        ## the state is the gameboard (4x4) grid and the current tile. It is probably simpler to get from this binary representation to a integer representation
        ## for this we first convert the gameboard to a binary representation. Flatten the gameboard and convert to binary
        flattenedStates = self.gameboard.board.flatten() 
        ## add the tile id
        tileId = self.gameboard.cur_tile_type

        self.tilebinary = self.tileIntegerToBinary(tileId)
        flattenedStates = np.concatenate((flattenedStates, self.tilebinary))
        self.stateId = self.binaryToInteger(flattenedStates)

        self.stateTracker.append(self.stateId)

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the Q-table or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        ## here we should implement the Q-learning algorithm

        self.actionIsTaken = False
        flag = 0
        taboList = []
        self.validationValue = 0
        while self.validationValue == 0:
    
            ## simply implement the gready policy
            if self.epsilon == 0:
                # print("state ID: ", self.stateId)
                # print("Q-table shape: ", self.Q.shape)
                
                ## ! we need to handle the case where we have same Q-values for multiple actions
                ## than we should choose a random of these actions

                if self.actionIsTaken == False and len(np.where(self.Q[self.stateId] == np.max(self.Q[self.stateId]))[0]) > 1 and len(taboList) < len(np.where(self.Q[self.stateId] == np.max(self.Q[self.stateId]))[0]):
                    # print("Multiple actions with the same Q-value")
                    
                    self.last_actionID = np.random.choice(np.where(self.Q[self.stateId] == np.max(self.Q[self.stateId]))[0])
                    # ! Here we should check if the action to be not in the tabo list -> future work
                    taboList.append(self.last_actionID)

                    self.actionIsTaken = True

                else:
                    flag = 1
                    self.last_actionID = np.argmax(self.Q[self.stateId])

                ## ! we should probably check if the action is valid
                ## how could the agent know if the action is valid?

                # execute the action
                self.validationValue = self.executeAction(self.last_actionID)

                # if self.validationValue == 0: we need to undo the action
                if (self.validationValue == 0 and flag == 0):
                    self.executeReverseAction(self.last_actionID)


            ## implement the epsilon greedy policy
            else:
                # check that epsilon is between 0 and 1
                if self.epsilon < 0 or self.epsilon > 1:
                    Exception("Epsilon should be between 0 and 1")
                
                ## generate a random number between 0 and 1
                randomValue = random.random()
                if randomValue < self.epsilon:
                    ## choose a random action
                    self.last_actionID = random.randint(0, self.numberOfActions-1)
                    validationValue = self.executeAction(self.last_actionID)

                    ## ? maybe we could use a more fancy random selection here, something like roulette wheel selection. But lets keep it simple for now
                
                else:
                    self.last_actionID = np.argmax(self.Q[self.stateId])
                    # execute the action
                    validationValue = self.executeAction(self.last_actionID)          



        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not
    
    def fn_reinforce(self,old_state,reward):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q table using state and action stored as attributes in self and using function arguments for the old state and the reward
        # This function should not return a value, the Q table is stored as an attribute of self

        ## here we should probably implement the equation 3 from the assignment
        ## $Q_{t+1}(s_{t},a_{t})=Q_{t}(s_{t},a_{t})+\alpha\left(r_{t+1}+\mathrm{max}_{a}Q_{t}(s_{t+1},a)-Q_{t}(s_{t},a_{t})\right)$
        # print(self.last_actionID)
        # print(self.Q[old_state])
        tmp1 = self.Q[old_state, self.last_actionID]
        tmp2 = self.alpha * (reward + np.max(self.Q[self.stateId]) - self.Q[old_state, self.last_actionID])
        # if (tmp1 + tmp2) < 0:
        #     print("Updated Q-value: ", tmp1 + tmp2)
        ## update the Q-table
        self.Q[old_state, self.last_actionID] = tmp1 + tmp2
        
        # Useful variables: 
        # 'self.alpha' learning rate

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1

            self.reward_tots = np.array(self.reward_tots)
            # determine the total reward for the episode
            self.totalrewards = np.sum(self.reward_tots)
            self.totalRewardTracker.append(self.totalrewards)
            self.gameLengthTracker.append(self.gameLength)

            if (self.totalrewards > self.bestReward):
                self.bestReward = self.totalrewards
                self.bestQ = self.Q
                self.bestStateTracker = self.stateTracker
                self.bestActionTracker = self.actionTracker
                print("New best reward: ", self.bestReward)
                print("=======================================")
                print("Episode: ", self.episode)
                print("Sum of Rewards in this episode: ", np.sum(self.reward_tots))
                print("Total reward for this episode: ", self.totalrewards)
                print("Total length of episode: ", self.gameLength)
                print("=======================================")

            # print(self.episode)
            if self.episode%1000==0:
                # print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots)),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    
                    # read logfiles/conf/counter.txt
                    now = datetime.datetime.now()
                    # index = now.strftime("%Y-%m-%d %H:%M")
                    

                    # # write index to /logfiles/conf/latest.txt
                    # with open("logfiles/conf/latest.txt", "w") as f:
                    #     f.write(index)

                    bestScore = self.readLogJSON("bestScore")
                    index = self.sessionIndex
                    
                    # if np.max(np.array(self.totalRewardTracker)) > bestScore:
                    #     self.writeBestRewardValue(self.totalrewards, "newBest", index)
                    #     print("New best score: ", self.totalrewards)
                    # else:
                    self.writeBestRewardValue(bestScore, "somelabel", index)

                    # save the rewards
                    np.save("logfiles/rewards_" + str(index) + '_' + str(self.episode), np.array(self.totalRewardTracker))
                    # save the Q-table
                    np.save("logfiles/Q-table_" + str(index)+ '_' + str(self.episode), self.bestQ)
                    # save game length tracker
                    np.save("logfiles/gameLengthTracker_" + str(index)+ '_' + str(self.episode), np.array(self.gameLengthTracker))
                    # save the state tracker
                    np.save("logfiles/stateTracker_" + str(index)+ '_' + str(self.episode), np.array(self.bestStateTracker))
                    # save the action tracker
                    np.save("logfiles/actionTracker_" + str(index)+ '_' + str(self.episode), np.array(self.bestActionTracker))
                    # # save the moving average tracker
                    # np.save("logfiles/movingAverageTracker_" + str(index)+ '_' + str(self.episode), np.array(self.movingAverageTracker))

                    # save the state tracker
                    # np.save("logfiles/stateTracker_" + str(index), np.array(self.stateTracker))
                    print("Saved as version: " + str(index))

                    
            if self.episode>=self.episode_count:
                # override log.json
                self.writeBestRewardValue(self.bestReward, "bestScore", self.sessionIndex)

                # save the best Q-table
                np.save("logfiles/bestQ-table_" + str(self.sessionIndex), self.bestQ)            

                raise SystemExit(0)
            else:
                self.reward_tots = []
                self.gameLength = 0
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.actionIsTaken = False
            self.fn_select_action()
            self.actionTracker.append(self.last_actionID)
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()
            old_state = self.stateId
            self.stateTracker.append(old_state)
            # Drop the tile on the game board and reveive the reward
            self.reward=self.gameboard.fn_drop()

            # print(self.reward)
            self.actionIsTaken = True
            self.gameLength += 1
            # print("Drop: ", self.reward)
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots.append(self.reward)

            # Read the new state
            self.fn_read_state()

            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state,self.reward)


    ## function to convert the binary state representation to an integer
    ## the binary representation is a array of 0s and 1s
    def binaryToInteger(self, binary):
        # the binary array comes in a 1D array
        binary = binary + 1
        binary = binary / 2

        # reverse the array
        binary = binary[::-1]

        value = 0
        for i in range(len(binary)):
            value += binary[i] * 2 ** i
        return int(value)
    
    def tileIntegerToBinary(self, tileID):
        if tileID == 0:
            return np.array([-1, -1])
        elif tileID == 1:
            return np.array([-1, 1])
        elif tileID == 2:
            return np.array([1, -1])
        elif tileID == 3:
            return np.array([1, 1])
        else :
            Exception("The tile id is not valid")

    def moveLeft(self):
        return self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)    

    def moveLeftLeft(self):
        validationValue0 = self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
        # print("LeftLeft val 0: ", validationValue0)
        validationValue1 = self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
        # print("LeftLeft val 1: ", validationValue1)
        # if one of the two is 0, returns 0 
        return validationValue0 * validationValue1

    def moveRight(self):
        return self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)

    def moveRightRight(self):
        validationValue = self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
        validationValue *= self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
        return validationValue
    
    # def moveDown(self):
    #     self.reward_tots[self.episode]+=self.gameboard.fn_drop()
        

    def rotate(self, numberOfRotations):
        for i in range(numberOfRotations):
            self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))

    def executeAction(self, actionID):
        # cases for the actionID
        ## 0   ->  [No Trans, No Rot]
        ## 1   ->  [No Trans, 90]
        ## 2   ->  [No Trans, 180]
        ## 3   ->  [No Trans, 270]
        ## 4   ->  [Left, No Rot]
        ## 5   ->  [Left, 90]
        ## 6   ->  [Left, 180]
        ## 7   ->  [Left, 270]
        ## 8   ->  [LeftLeft, No Rot]
        ## 9   ->  [LeftLeft, 90]
        ## 10  ->  [LeftLeft, 180]
        ## 11  ->  [LeftLeft, 270]
        ## 12  ->  [Right, No Rot]
        ## 13  ->  [Right, 90]
        ## 14  ->  [Right, 180]
        ## 15  ->  [Right, 270]
        validationValue = 1
        match actionID:
            case 0:                 # [No Trans, No Rot]
                pass
            case 1:                 # [No Trans, 90]
                self.rotate(1)
            case 2:                 # [No Trans, 180]
                self.rotate(2)
            case 3:                 # [No Trans, 270]
                self.rotate(3)
            case 4:                 # [Left, No Rot]
                validationValue = self.moveLeft()
            case 5:                 # [Left, 90]
                self.rotate(1)
                validationValue = self.moveLeft()
            case 6:                 # [Left, 180]
                self.rotate(2)
                validationValue = self.moveLeft()    
            case 7:                 # [Left, 270]
                self.rotate(3)
                validationValue = self.moveLeft()                
            case 8:                 # [LeftLeft, No Rot]    
                validationValue = self.moveLeftLeft()
            case 9:                 # [LeftLeft, 90]
                self.rotate(1)
                validationValue = self.moveLeftLeft()
            case 10:                # [LeftLeft, 180]
                self.rotate(2)
                validationValue = self.moveLeftLeft()    
            case 11:                # [LeftLeft, 270]
                self.rotate(3)
                validationValue = self.moveLeftLeft()
            case 12:                # [Right, No Rot]
                validationValue = self.moveRight()
            case 13:                # [Right, 90]
                self.rotate(1)
                validationValue = self.moveRight()       
            case 14:                # [Right, 180]
                self.rotate(2)
                validationValue = self.moveRight()
            case 15:                # [Right, 270]
                self.rotate(3)
                validationValue = self.moveRight()
                
            case _:                 # default
                Exception("The action ID is not valid")
        self.validationValue = validationValue
        return validationValue
    
    def executeReverseAction(self, actionID):
        match actionID:
            case 0:
                pass
            case 1:
                self.rotate(3)
            case 2:
                self.rotate(2)
            case 3:
                self.rotate(1)
            case 4:
                self.moveRight()
            case 5:
                self.rotate(3)
                self.moveRight()
            case 6:
                self.rotate(2)
                self.moveRight()
            case 7:
                self.rotate(1)
                self.moveRight()
            case 8:
                self.moveRightRight()
            case 9:
                self.rotate(3)
                self.moveRightRight()
            case 10:
                self.rotate(2)
                self.moveRightRight()
            case 11:
                self.rotate(1)
                self.moveRightRight()
            case 12:
                self.moveLeft()
            case 13:
                self.rotate(3)
                self.moveLeft()
            case 14:
                self.rotate(2)
                self.moveLeft()
            case 15:
                self.rotate(1)
                self.moveLeft()
            case _:
                Exception("The action ID is not valid")
            
    def readLogJSON(self, key):
        with open("logfiles/conf/log.json", "r") as file:
            data = json.load(file)
            return (data[key])
    
    def writeBestRewardValue(self, value, label="Placeholder", index=000000):
        data = {"bestScore": int(value), "bestScoreLabel": label, "index": int(index)}
        print(data)
        with open("logfiles/conf/log.json", "w") as file:
            json.dump(data, file)

class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilonInit = epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count


    # function to intialize the neural network
    def fn_init_NN(self, numberOfHiddenNeurons):
        self.Q_net = torch.nn.Sequential(
            torch.nn.Linear(self.numberOfStates, numberOfHiddenNeurons),
            torch.nn.ReLU(),
            torch.nn.Linear(numberOfHiddenNeurons, numberOfHiddenNeurons),
            torch.nn.ReLU(),
            torch.nn.Linear(numberOfHiddenNeurons, self.numberOfActions)
        )
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=self.alpha) #? I guess that alpha is the learning rate


        self.Q_target_net = torch.nn.Sequential(
            torch.nn.Linear(self.numberOfStates, numberOfHiddenNeurons),
            torch.nn.ReLU(),
            torch.nn.Linear(numberOfHiddenNeurons, numberOfHiddenNeurons),
            torch.nn.ReLU(),
            torch.nn.Linear(numberOfHiddenNeurons, self.numberOfActions)
        )

        print(self.Q_net)
        print(self.Q_target_net)


    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self
        
        ## Init number of states and actions, need to init the Networks
        self.numberOfStates = gameboard.N_row * gameboard.N_col + 2;
        self.numberOfActions = 16
   
        ## Init the Networks
        self.numberOfHiddenNeurons = 64
        self.fn_init_NN(self.numberOfHiddenNeurons)
        self.gameReward = 0
        self.gameRewardTracker = []
        self.gameLength = 0
        self.gameLengthTracker = []
        
        ## init the buffer for the replay data
        self.buffer = []
        
        ## get the first state :D 
        self.fn_read_state()

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self,strategy_file):
        self.Q_net.load_state_dict(torch.load(strategy_file))


        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self
        self.state = self.gameboard.board.copy()
        self.tileState = self.gameboard.cur_tile_type

        # handle initial tile state
        if self.tileState == -1:
            self.tileState = 0

        ## hardcode the tile state
        match self.tileState:
            case 0:
                self.tileState = np.array([0, 0])
            case 1:
                self.tileState = np.array([0, 1])
            case 2:
                self.tileState = np.array([1, 0])
            case 3:
                self.tileState = np.array([1, 1])

        self.tileState *= 2
        self.tileState -= 1

        self.state = np.concatenate((self.state.flatten(), self.tileState))

        self.state += 1
        self.state /= 2

        # print(self.state)

        # convert the state to a tensor
        self.state = torch.tensor(self.state, dtype=torch.float32)


        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):

        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        ## greefy policy
        r = np.random.random()
        ## update the eosilon
        self.validationValue = 0
    # while self.validationValue == 0:
        self.epsilon = self.fn_getEpsilon()
        if r < self.epsilon:
            self.action = np.random.randint(0, self.numberOfActions)
        else:
            self.action = torch.argmax(self.Q_net(self.state)).item()

        ## execute the action
        newX = self.action // 4
        newOrient = self.action % 4

        self.validationValue = self.gameboard.fn_move(newX, newOrient)

            # print("Action:", newX, newOrient)


        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self,batch):
        
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        states, next_states = [], []
        for sample in batch:
            states.append(sample["old_state"])
            next_states.append(sample["new_state"])

        targets = torch.zeros(self.batch_size, self.numberOfActions)
        # now we need a mask, as in our replay data we have for a given state only one action/reward
        targets_mask = torch.zeros(self.batch_size, self.numberOfActions)

        # now get the approximation from the target network
        with torch.no_grad():
            q_hat = self.Q_target_net(torch.stack(next_states, dim=0))
        
        # compute the targets
        for index, sample in enumerate(batch):
            if sample["gameover"]:
                y = sample["reward"]
            else:
                y = sample["reward"] + np.nanmax(q_hat[index, :])
            targets[index, sample["action"]] = y
            targets_mask[index, sample["action"]] = 1

        # Evaluate the old states, apply the mask and update the weights
        self.optimizer.zero_grad()
        outputs = self.Q_net(torch.stack(states, dim=0)) * targets_mask
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()


        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1

            self.gameRewardTracker.append(self.gameReward)
            self.gameLengthTracker.append(self.gameLength)

          
            self.gameReward = 0


            if self.episode%100==0:
                gameRewardTrackerCopy = np.array(self.gameRewardTracker)
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(gameRewardTrackerCopy[range(self.episode-100,self.episode)])/100),')')
                # self.print_network_weights(self.Q_net)
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    pass
                    # # TO BE COMPLETED BY STUDENT
                    # # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                self.sessionIndex = "DQN_testFreedAxl"
                #self.writeBestRewardValue(self.bestReward, "bestScore", self.sessionIndex)
                np.save("logfiles/rewards_DQ" + str(self.sessionIndex) + '_' + str(self.episode), np.array(self.gameRewardTracker))
                np.save("logfiles/gameLengthTracker_DQ" + str(self.sessionIndex)+ '_' + str(self.episode), np.array(self.gameLengthTracker))
                # save the Q-net
                torch.save(self.Q_net.state_dict(), "logfiles/Q-net_" + str(self.sessionIndex)+ '_' + str(self.episode))

                raise SystemExit(0)
            
            self.gameboard.fn_restart()

        
            if (len(self.buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                pass
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to copy the current network to the target network
                self.Q_target_net.load_state_dict(self.Q_net.state_dict())
                print("Synced the networks")

            else:
                self.gameboard.fn_restart()
        else:
            
            # feed the replay 
            old_state = self.state


            with torch.no_grad():

                # Select and execute action (move the tile to the desired column and orientation)
                self.fn_select_action()
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

                # Drop the tile on the game board
                reward = self.gameboard.fn_drop()
                self.gameReward += reward

                # print(reward)

                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

                # Read the new state
                self.fn_read_state()

                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to store the state in the experience replay buffer
                self.buffer.append(
                {
                    "old_state": old_state,
                    "action": self.action,
                    "reward": reward,
                    "new_state": self.state,
                    "gameover": self.gameboard.gameover,
                })
                if (len(self.buffer) >= self.replay_buffer_size + 1):
                    self.buffer.pop(0)
                

            if len(self.buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                batch = random.sample(self.buffer, self.batch_size)              
                
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                self.fn_reinforce(batch)

    def fn_getEpsilon(self):
        tmp = np.array([self.epsilonInit, 1 - self.episode / self.epsilon_scale])
        # print(tmp)
        tmp = np.max(tmp)

        return tmp
        # return self.epsilon

    def readLogJSON(self, key):
        with open("logfiles/conf/log.json", "r") as file:
            data = json.load(file)
            return (data[key])
    
    def writeBestRewardValue(self, value, label="Placeholder", index=000000):
        data = {"bestScore": int(value), "bestScoreLabel": label, "index": int(index)}
        print(data)
        with open("logfiles/conf/log.json", "w") as file:
            json.dump(data, file)

    def print_network_weights(self, network):
        for name, param in network.named_parameters():
            if param.requires_grad:
                print(name, param.data)


class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()