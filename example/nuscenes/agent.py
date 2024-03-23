from pgeon import Agent
from collections import deque
import numpy as np
REPLAY_MEMORY_SIZE = 5_000


class SelfDrivingAgent(Agent):
    def __init__(self, observation_space, action_space):#, path):
        #self.agent = # Algorithm.from_checkpoint(path)
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, state):
        # Implement your action selection logic here, 
        # e.g., using a deep neural network or other policy 
        # based on the provided state information.
        # Replace the following with your actual implementation.
        action = np.random.choice(self.action_space)
        return action
        #return self.agent.compute_single_action(state)
    
