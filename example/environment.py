from typing import Optional, Any, Tuple
from pgeon.environment import Environment
import numpy as np
from example.av_discretizer import Position, Velocity, Action, Rotation, AVDiscretizer
from pgeon.discretizer import Predicate


class SelfDrivingEnvironment(Environment):
    environment_name = "NuScenes Environemnt"

    def __init__(self):

        self.current_state = None
        self.discretizer = AVDiscretizer()

    # TODO Set seed, introduce randomness
    def reset(self, seed: Optional[int] = None) -> Any:
        self.current_state = [0,0,0,0]#(Position(0, 0), Velocity.STOPPED, Rotation.FORWARD)


    def step(self, action:Action) -> Tuple[Tuple[Predicate, Predicate, Predicate], float, bool]:
        # Given the current state and the chosen action, determine the next state and the reward
        reward = 0.1  
        next_state = self.apply_action(action, self.discretizer.discretize(self.current_state))
        #TODO: should the next state be createed by me or retrieved by the MDP?
        reward = self.compute_reward(next_state)       

        # Check for terminal conditions (e.g., collision, reaching destination)
        done = False
        extra_info = {} 
        return next_state, reward, done, extra_info
    

    def apply_action(self, action:Action):

        pass


    @staticmethod
    def compute_reward(current_state, action, next_state, is_destination):
        """
        Computes the reward for transitioning from the current_state to next_state via action.

        Args:
            current_state: discretized current velocity, position and rotation action,
            action:
            next_state: discretized next velocity, position and rotation action
            is_destination: True is next_state is a final state, False otherwise.

        Return:
            float: final reward
        """

        reward = 0

        velocity_predicate = current_state[1].value[0]
        velocity = Velocity[str(velocity_predicate)[:-1].split('(')[1]]
        
        next_velocity_predicate = next_state[1].value[0]
        next_velocity = Velocity[str(next_velocity_predicate)[:-1].split('(')[1]]
       

        # Encourage maintaining a safe and moderate speed
        if velocity == Velocity.MEDIUM:
            reward += 1
        elif velocity in [Velocity.LOW, Velocity.HIGH]:
            reward += 0.5
        elif velocity == Velocity.VERY_HIGH:
            reward -= 1  # Penalize very high speeds for safety reasons

        # Penalize stopping in potentially unsafe or unnecessary situations
        if velocity == Velocity.STOPPED:
            reward -= 0.5

        # Encourage smooth driving: penalize sudden actions that might indicate aggressive driving
        if action in [Action.GAS, Action.BRAKE]:
            reward -= 0.2

        # Encourage staying straight unless necessary to turn
        if action in [Action.TURN_LEFT, Action.TURN_RIGHT]:
            reward -= 0.1
        elif action == Action.STRAIGHT:
            reward += 0.2

        # If considering the next state, encourage actions leading to safer states
        if next_state:
            if next_velocity in [Velocity.MEDIUM, Velocity.LOW]:
                reward += 0.5
            if next_velocity == Velocity.STOPPED:
                reward -= 0.5  # Assuming stopping might not always be ideal

        # Encourage actions that lead to progress towards a goal
        if is_destination:
            reward += 10

        return reward



    #@staticmethod
    def compute_total_reward(self, agent, initial_state, final_state, max_steps=100):
        """
        Computes the total reward obtained by following a policy from an initial state to a final state.
        
        Args:
        - policy: The policy to follow.
        - initial_state: The state from which to start (continuos).
        - v: final state.
        - max_steps: Maximum number of steps to prevent infinite loops.
        
        Returns:
        - total_reward: The total reward obtained.
        - reached_final: Boolean indicating if the final state was reached.
        """
        total_reward = 0
        step_count = 0
        reached_final = False

        self.current_state = initial_state

        while step_count < max_steps:
            action = agent.act(self.current_state)
            next_state, reward, done, _ = self.step(action)
            total_reward +=reward
            step_count +=1

            if (next_state == final_state).all(): #TODO: review
                reached_final = True
                break
            elif done:
                # If 'done' is True, but the final state hasn't been reached, the episode ended unexpectedly.
                break

            self.current_state = next_state

        return total_reward, reached_final
        
    