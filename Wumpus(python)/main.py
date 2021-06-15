# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 23:34:44 2021

@author: Asus
"""

import numpy as np

# =============================================================================
# definition of environment and inputs
# =============================================================================
#the shape of the environment
environment_rows = 5 
environment_columns = 5

start = (0, 0) #start point

hole1 = (2, 0) #hole1 point
hole2 = (2, 1) #hole2 point
wall = (3, 2) #wall point

goal = (4, 0) #goal point

episodes = 500 #one sequence of states, actions, and rewards, which ends with a terminal state

epsilon = 0.5 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.8 #discount factor for future rewards

# =============================================================================
# Q-Learning class
# =============================================================================
class Q_Learning:
    def __init__(self, environment_rows, environment_columns, start, hole1, hole2, wall, goal, episodes, epsilon, discount_factor):
        self.environment_rows = environment_rows
        self.environment_columns = environment_columns
        self.start = start
        self.hole1 = hole1
        self.hole2 = hole2
        self.wall = wall
        self.goal = goal
        self.episodes = episodes
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        
    def set_matrices(self):
        #create a 2D numpy array to hold the rewards for each state
        self.rewards = np.full((self.environment_rows, self.environment_columns), -1)
        self.rewards[self.hole1] = self.rewards[self.hole2] = self.rewards[self.wall] = -100
        self.rewards[self.goal] = 100
        
        #create a 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
        self.q_values = np.zeros((self.environment_rows, self.environment_columns, 4))

        #define actions
        #numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
        self.actions = ['up', 'right', 'down', 'left']
    
    #define a function that determines if the specified location is a terminal state
    def is_terminal_state(self, current_row_index, current_column_index):
        if self.rewards[current_row_index, current_column_index] == -1:
            return False
        else:
            return True

    #define an epsilon greedy algorithm that will choose which action to take next
    def get_next_action(self, current_row_index, current_column_index, epsilon):
        #if a randomly chosen value between 0 and 1 is less than epsilon, 
        #then choose the most promising value from the Q-table for this state.
        if np.random.random() < epsilon:
            return np.argmax(self.q_values[current_row_index, current_column_index])
        else: #choose a random action
            return np.random.randint(4)

    #define a function that will get the next location based on the chosen action
    def get_next_location(self, current_row_index, current_column_index, action_index):
        new_row_index = current_row_index
        new_column_index = current_column_index
        if self.actions[action_index] == 'up' and current_row_index > 0:
            new_row_index -= 1
        elif self.actions[action_index] == 'right' and current_column_index < self.environment_columns - 1:
            new_column_index += 1
        elif self.actions[action_index] == 'down' and current_row_index < self.environment_rows - 1:
            new_row_index += 1
        elif self.actions[action_index] == 'left' and current_column_index > 0:
            new_column_index -= 1
        return new_row_index, new_column_index


    #run through 500 training episodes
    def train(self):
        for episode in range(self.episodes):
            #get the starting location for this episode
            row_index, column_index = self.start

            #continue taking actions (i.e., moving) until we reach a terminal state
            #(i.e., until we reach the item packaging area or crash into an item storage location)
            while not self.is_terminal_state(row_index, column_index):
      
                #choose which action to take (i.e., where to move next)
                action_index = self.get_next_action(row_index, column_index, self.epsilon)

                #store the old row and column indexes
                old_row_index, old_column_index = row_index, column_index 
    
                #perform the chosen action, and transition to the next state (i.e., move to the next location)
                row_index, column_index = self.get_next_location(row_index, column_index, action_index)
    
                #receive the reward for moving to the new state, and calculate the temporal difference
                reward = self.rewards[row_index, column_index]
                new_q_value = reward + (self.discount_factor * np.max(self.q_values[row_index, column_index]))

                #update the Q-value for the previous state and action pair
                self.q_values[old_row_index, old_column_index, action_index] = new_q_value


    def get_shortest_path(self, start):
        start_row_index, start_column_index = start
        #return immediately if this is an invalid starting location
        if self.is_terminal_state(start_row_index, start_column_index):
            return []
        else: #if this is a 'legal' starting location
            current_row_index, current_column_index = start_row_index, start_column_index
            shortest_path = []
            shortest_path.append([current_row_index, current_column_index])
            #continue moving along the path until we reach the goal (i.e., the item packaging location)
            while not self.is_terminal_state(current_row_index, current_column_index):
                #get the best action to take
                action_index = self.get_next_action(current_row_index, current_column_index, 1.)
                #move to the next location on the path, and add the new location to the list
                current_row_index, current_column_index = self.get_next_location(current_row_index, current_column_index, action_index)
                shortest_path.append([current_row_index, current_column_index])
        return shortest_path
    
# =============================================================================
#     
# =============================================================================
Q_L = Q_Learning(environment_rows, environment_columns, start, hole1, hole2, wall, goal, episodes, epsilon, discount_factor)
Q_L.set_matrices()

Q_L.rewards

Q_L.train()
print("finished training")

print(Q_L.get_shortest_path(start))