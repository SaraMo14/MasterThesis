from typing import Optional, Any, Tuple
from pgeon.environment import Environment
from example.discretizer.utils import Velocity, Action, Rotation
from example.discretizer.discretizer import AVDiscretizer
from example.agent import SelfDrivingAgent
from pgeon.discretizer import Predicate


class SelfDrivingEnvironment(Environment):
    environment_name = "nuScenes Environemnt"

    def __init__(self, discretizer:AVDiscretizer):

        self.current_state = None
        self.discretizer = discretizer
        self.WEIGHT_SPEED = 1
        self.WEIGHT_SAFETY = 2
        self.WEIGHT_SMOOTHNESS = 1
        self.WEIGHT_PROGRESS = 1


        self.agent = SelfDrivingAgent()

    # TODO Set seed, introduce randomness and fix initialization
    def reset(self, seed: Optional[int] = None) -> Any:
        self.current_state = Predicate(0,0,0,0)


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
        
        #update car state
        next_state = self.agent.act(action)
        
        # compute reward
        reward = self.compute_reward(action,is_destination )       

        # check if the episode is done

        return next_state, reward, False, None
    

    '''
    def act(self, action: Action):
        """
        Function that given current state and action returns the next continuous state.
        ref: https://es.mathworks.com/help/mpc/ug/obstacle-avoidance-using-adaptive-model-predictive-control.html

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
      
    def compute_reward(self,action, is_destination):
        reward = 0
    
        # Penalty for exceeding speed limit
        speed_penalty = min(speed, 10 - speed)
        reward += speed_penalty
        
        # Penalty for steering angle
        steering_penalty = -0.5 * (steering_angle ** 2)
        reward += steering_penalty
    
        # Penalty for collision
        if collision:
            reward -= 10
        
        # Penalty for leaving the lane
        #if distance_to_lane > 2:
        #    reward -= 1
        
        # Encourage reaching the goal quickly
        reward -= 0.1
        
        # Additional reward upon reaching the goal
        if reached_goal:
            reward += 10  # Or any positive value to signify reaching the goal
        
        return reward


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
        #Ref: https://arxiv.org/pdf/2405.01440
        #TODO: penalize progress away from the goal, reward progress toward the goal
        #TODO: rewards between -1,0,1
        # Initialize reward components
        speed_reward = 0
        safety_reward = 0
        smoothness_reward = 0
        progress_reward = 0

        #TODO: modify to handle discretized state
        vel_predicate = self.current_state[1].value[0]
        velocity = Velocity[str(vel_predicate)[:-1].split('(')[1]]
        yaw_predicate = self.current_state[2].value[0]
        yaw = Velocity[str(yaw_predicate)[:-1].split('(')[1]]
        
        # Encourage maintaining a safe and moderate speed
        if velocity in [Velocity.LOW, Velocity.MEDIUM]:
            speed_reward += 0.1
        elif velocity in [Velocity.HIGH, Velocity.VERY_HIGH]:
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

        # To encourage actions that lead to progress towards a goal, give positive reward if current state is a intermediate destination
        if is_destination:
            progress_reward += 0.5
        else:
            progress_reward -=0.1


        
        total_reward = (
            speed_reward * self.WEIGHT_SPEED
            + safety_reward * self.WEIGHT_SAFETY
            + smoothness_reward * self.WEIGHT_SMOOTHNESS
            + progress_reward * self.WEIGHT_PROGRESS
        )

        return total_reward#, speed_reward, safety_reward, smoothness_reward, progress_reward
        #TODO: add final goal rewad.

