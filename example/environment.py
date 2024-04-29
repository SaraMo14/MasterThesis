from typing import Optional, Any, Tuple
from pgeon.environment import Environment
import numpy as np
from example.av_discretizer_complex import Velocity, Action, Rotation, AVDiscretizer
from pgeon.discretizer import Predicate


class SelfDrivingEnvironment(Environment):
    environment_name = "nuScenes Environemnt"

    def __init__(self):

        self.current_state = None
        self.discretizer = AVDiscretizer()
        self.WEIGHT_SPEED = 1
        self.WEIGHT_SAFETY = 2
        self.WEIGHT_SMOOTHNESS = 1
        self.WEIGHT_PROGRESS = 1

    # TODO Set seed, introduce randomness
    def reset(self, seed: Optional[int] = None) -> Any:
        self.current_state = np.array([0,0,0,0]) #Predicate()


    def step(self, action:Action, is_destination) -> Tuple[Tuple[Predicate, Predicate, Predicate], float, bool]:
        """
        Perform a step in the environment based on the computation of P(s'|s,a) as #(s,a,s')/(s,a).

        Args:
            action (Action): Action to be taken.
            is_destination: True is current state is a (intermediate) destination state, False otherwise

        Returns:
            next_state: tuple of predicates representing the next state after taking the action.
            reward: 
        """
         
        next_state = self.apply_action(action)
        reward = self.compute_reward(action,is_destination )       

        return next_state, reward, False, None
    
    
    #ref: https://es.mathworks.com/help/mpc/ug/obstacle-avoidance-using-adaptive-model-predictive-control.html
    '''
    def apply_action(self, action: Action):
        """
        Function that given current state and action returns the next continuous state
        """
        x, y, v, yaw = self.current_state
        delta_v = self.discretizer.frequency * (self.discretizer.eps_vel*2)
        delta_y = v * np.sin(yaw) * self.discretizer.frequency  
        delta_x = v * np.cos(yaw) * self.discretizer.frequency 
        

        if action == Action.REVERSE:
            next_state = np.array([x, y - delta_y,v, yaw])
        elif action == Action.STRAIGHT:
            next_state = np.array([x, y + delta_y, v, yaw])
        elif action == Action.BRAKE:
            next_state = np.array([x, y + delta_y, v - delta_v, yaw])
        elif action == Action.GAS:
            next_state = np.array([x, y + delta_y, v + delta_v, yaw])
        elif action == Action.TURN_RIGHT:
            next_state = np.array([x + delta_x, y, v, yaw+self.discretizer.eps_rot*2])
        elif action == Action.TURN_LEFT:
            next_state = np.array([x - delta_x, y, v, yaw-self.discretizer.eps_rot*2])
        elif action == Action.GAS_TURN_LEFT:
            next_state = np.array([x - delta_x, y + delta_y, v + delta_v, yaw-self.discretizer.eps_rot*2])
        elif action == Action.GAS_TURN_RIGHT:
            next_state = np.array([x + delta_x, y + delta_y, v + delta_v, yaw+self.discretizer.eps_rot*2])
        elif action == Action.BRAKE_TURN_LEFT:
            next_state = np.array([x - delta_x, y + delta_y, v - delta_v, yaw-self.discretizer.eps_rot*2])
        elif action == Action.BRAKE_TURN_RIGHT:
            next_state = np.array([x + delta_x, y + delta_y, v - delta_v, yaw+self.discretizer.eps_rot*2])
        else:
            # for IDLE action or any undefined action, return current state
            next_state = self.current_state
            #TODO: fix since when it arrives at idle, the state remains unchanged
            #you should add a very small epsilon maybe, just to change a bit the state so it doesnt get stuck.

        return np.array(next_state)
    '''
      
    def compute_reward_disc(self, action, is_destination):
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

        # Initialize reward components
        speed_reward = 0
        safety_reward = 0
        smoothness_reward = 0
        progress_reward = 0

        vel_predicate = self.current_state[1].value[0]
        velocity = Velocity[str(vel_predicate)[:-1].split('(')[1]]
        yaw_predicate = self.current_state[2].value[0]
        yaw = Velocity[str(yaw_predicate)[:-1].split('(')[1]]
        
        # Encourage maintaining a safe and moderate speed
        if velocity in [Velocity.LOW, Velocity.MEDIUM]:
            speed_reward += 0.1
        elif velocity == [Velocity.HIGH, Velocity.VERY_HIGH]:
            speed_reward -= 0.5 # Penalize very high speeds for safety reasons

        # Penalize stopping in potentially unsafe or unnecessary situations
        if velocity == Velocity.STOPPED:
            safety_reward -= 0.5

        # Encourage smooth driving: penalize sudden actions that might indicate aggressive driving
        if action in [Action.GAS, Action.BRAKE]:
            smoothness_reward -= 0.2

        # Encourage staying straight unless necessary to turn
        if action in [Action.TURN_LEFT, Action.TURN_RIGHT, Action.GAS_TURN_LEFT, Action.GAS_TURN_RIGHT]:
            smoothness_reward -= 0.1
        elif action == Action.STRAIGHT:
            smoothness_reward += 0.2

        if yaw in [Rotation.LEFT, Rotation.RIGHT]:
            smoothness_reward -=0.5
        elif yaw in [Rotation.SLIGHT_LEFT, Rotation.SLIGHT_RIGHT]:
            smoothness_reward -=0.2
        else:
            smoothness_reward +=0.1

        # Encourage actions that lead to progress towards a goal.
        # Give positive reward if current state is a intermediate destination
        if is_destination:
            progress_reward += 10

        
        total_reward = (
            speed_reward * self.WEIGHT_SPEED
            + safety_reward * self.WEIGHT_SAFETY
            + smoothness_reward * self.WEIGHT_SMOOTHNESS
            + progress_reward * self.WEIGHT_PROGRESS
        )

        return total_reward#, speed_reward, safety_reward, smoothness_reward, progress_reward
        


    #@staticmethod
    #TODO:  update
    def compute_total_reward(self, agent, initial_state, final_state, max_steps=600):
        """
        Computes the total reward obtained by following a policy from an initial state to a final state.
        
        Args:
        - agent: The agent following the policy.
        - initial_state: The state from which to start (continuos).
        - final_state: The final state to reach.
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
            action_id, is_destination = agent.act(self.current_state)
            action = self.discretizer.get_action_from_id(action_id)
            print(action) 
            next_state, reward, _, _ = self.step(action, is_destination)
            total_reward +=reward
            step_count +=1

            print(step_count)
            self.current_state = next_state
            if np.array_equal(next_state,final_state):
                reached_final = True
                break

        return total_reward, reached_final
        
    