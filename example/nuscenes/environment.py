import gym
from typing import Optional, Any, Tuple
from pgeon.environment import Environment
import numpy as np



class SelfDrivingEnvironment(Environment):

    def __init__(self, sensor_dim = 0, history_length=0, action_space_low=0, action_space_high=0):
        self.sensor_dim = sensor_dim
        self.history_length = history_length
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        #action space
        
        #observation space 
        
        # (includes history buffer)
        
        # Initialize history buffer and state
        #self.sensor_history = np.zeros((self.history_length, self.sensor_dim))
        #self.state = None



    # TODO Set seed
    def reset(self, seed: Optional[int] = None) -> Any:
        # Reset environment and history buffer
        self.sensor_history = np.zeros((self.history_length, self.sensor_dim))
        self.state = {
            "sensor_data": self.sensor_history.copy(),
            "ego_pose": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # Example
        }
        return self.state

    def step(self, action) -> Tuple[Any, Any, bool, Any]:

        # Take action in the environment and get the next state and reward
        # (replace with actual simulation or interaction)
        new_sensor_data = np.random.rand(self.sensor_dim)  # Example sensor data
        reward = 0.1  # Example reward for staying in lane
        # Update history buffer
        self.sensor_history[:-1] = self.sensor_history[1:]
        self.sensor_history[-1] = new_sensor_data
        

        # Update state
        new_state = {
            "sensor_data": self.sensor_history.copy(),
            "ego_pose": self.state["ego_pose"] + np.random.rand(7) * 0.1,  # Example update
        }


        # Check for terminal conditions (e.g., collision, reaching destination)
        done = False
        info = {}  # Additional information for debugging
        return new_state, reward, done, info