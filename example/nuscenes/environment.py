from typing import Optional, Any, Tuple
from pgeon.environment import Environment
import numpy as np
from example.nuscenes.av_discretizer import Action, Velocity


class SelfDrivingEnvironment(Environment):
    environment_name = "NuScenes Environemnt"

    def __init__(self):

        #self.action_space = AVDiscretizer.all_actions()
        self.observation_space = ''
        self.position = 0,0
      

        return None

    # TODO Set seed
    def reset(self, seed: Optional[int] = None) -> Any:

        self.state = {
            "x": 0,
            "y": 0, 
            "velocity": 0,
            "yaw": 0,
            "acceleration": 0
        }
        return self.state


    def step(self, action) -> Tuple[Any, Any, bool, Any]:

        reward = 0.1  
        # Update state
        new_state = {
            "x": 0,
            "y": 0, 
            "velocity": 0,
            "yaw": 0,
            "acceleration": 0
        }


        # Check for terminal conditions (e.g., collision, reaching destination)
        done = False
        extra_info = {} 
        return new_state, reward, done, extra_info
    


    def compute_reward(self, current_state, action, next_state=None):
        """
        Computes the reward for transitioning from the current_state to next_state via action.

        Parameters:
        - current_state: The current state of the agent.
        - action: The action taken by the agent.
        - next_state: The state resulting from the action.

        Returns:
        - A numeric reward value.
        """

        reward = 0

        velocity_predicate = current_state[1].value[0]
        velocity = Velocity[str(velocity_predicate)[:-1].split('(')[1]]
        
        next_velocity_predicate = current_state[1].value[0]
        next_velocity = Velocity[str(next_velocity_predicate)[:-1].split('(')[1]]
       

        # Encourage maintaining a safe and moderate speed
        if velocity == Velocity.MEDIUM:
            reward += 10
        elif velocity in [Velocity.LOW, Velocity.HIGH]:
            reward += 5
        elif velocity == Velocity.VERY_HIGH:
            reward -= 10  # Penalize very high speeds for safety reasons

        # Penalize stopping in potentially unsafe or unnecessary situations
        if velocity == Velocity.STOPPED:
            reward -= 5

        # Encourage smooth driving: penalize sudden actions that might indicate aggressive driving
        if action in [Action.GAS, Action.BRAKE]:
            reward -= 2

        # Encourage staying straight unless necessary to turn
        if action in [Action.TURN_LEFT, Action.TURN_RIGHT]:
            reward -= 1
        elif action == Action.STRAIGHT:
            reward += 2

        # If considering the next state, encourage actions leading to safer states
        if next_state:
            if next_velocity in [Velocity.MEDIUM, Velocity.LOW]:
                reward += 5
            if next_velocity == Velocity.STOPPED:
                reward -= 5  # Assuming stopping might not always be ideal

        # Encourage actions that lead to progress towards a goal
        #if next_state and goal_achieved(next_state):
        #    reward += 100

        return reward



        
    def compute_and_store_rewards(self):
        #TODO: modify
        """
        Computes the cumulative reward for each trajectory.

        Parameters:
        - trajectories: A list of trajectories, each trajectory is a list of (current_state, action, next_state) tuples.
        
        Return:

        """
        for node in self.nodes:
            for neighbor in self[node]:
                for action, attrs in self[node][neighbor].items():
                    #current_state = self.discretizer.state_to_str(node)
                    #next_state = self.discretizer.state_to_str(neighbor)
                    reward = self.compute_reward(0, action, 0)
                    attrs['reward'] = reward
