import gym
from typing import Optional, Any, Tuple
from pgeon import Environment
import numpy as np

SECONDS_PER_EPISODE = 10

THROTTLE_MIN = 0.0
STEER_MIN= -1.0
THROTTLE_MAX = 1.0
STEER_MAX = 1.0

class SelfDrivingEnvironment(Environment):

    def __init__(self, sensor_dim, history_length, action_space_low, action_space_high):
        self.sensor_dim = sensor_dim
        self.history_length = history_length
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        #action space
        self.action_space = gym.spaces.Box(np.array([THROTTLE_MIN, STEER_MIN]), 
                          np.array([THROTTLE_MAX, STEER_MAX]), dtype=np.float32)  # Example action space
       
        #observation space 
       self.observation_space = gym.spaces.Box(low=0, high=255,
                        shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
        '''
        # (includes history buffer)
        self.observation_space = gym.spaces.Dict(
            {
                "sensor_data": gym.spaces.Box(
                    low=0, high=1, shape=(self.history_length, self.sensor_dim)
                ),
                "ego_pose": gym.spaces.Box(low=-10, high=10, shape=(7,))  # Example
                       #    "lidar_data": gym.spaces.Box(low=0, high=1, shape=(100, 3)),

            }
            
        )'''

        # Initialize history buffer and state
        self.sensor_history = np.zeros((self.history_length, self.sensor_dim))
        self.state = None



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

        '''
        if action == 0:
            #self.vehichle
            pass
        elif action == 1:
            pass
        elif action == 2:
            pass
        elif action == 3:
            pass
        v = self.vehicle.get_velocity()
        kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        if len(self.collision_hist) != 0:
            done = True
            reward = -200 
        elif kmh < 50:
            done = False
            reward = -1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None  #obs, reward, done, extra_info
        '''
        return new_state, reward, done, info


# Example usage
if __name__ == "__main__":
    sensor_dim = 100  # Example sensor dimension
    history_length = 3  # Example history length
    action_space_low = -1  # Example action space lower bound
    action_space_high = 1  # Example action space upper bound

    env = SelfDrivingEnvironment(
        sensor_dim, history_length, action_space_low, action_space_high
    )
    state = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}")
        state = next_state