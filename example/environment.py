from typing import Optional, Any, Tuple
from pgeon.environment import Environment
from example.discretizer.utils import Velocity, Action, Rotation
#from example.discretizer.discretizer import AVDiscretizer
from pgeon.discretizer import Predicate


class SelfDrivingEnvironment(Environment):

    def __init__(self): #discretizer:AVDiscretizer): 
        #self.city = "boston-seaport" #TODO: distinguish different cities
        
        #self.current_state = None
        #self.discretizer = discretizer
        self.WEIGHT_SPEED = 1
        self.WEIGHT_SAFETY = 2
        self.WEIGHT_SMOOTHNESS = 1
        self.WEIGHT_PROGRESS = 1


        self.epsilon = 1e-6 #small pertubation for idle states


    def reset(self, seed: Optional[int] = None) -> Any:
        #TODO: fix        
        #np.random.seed(seed)
        #start_node = np.random.choice(list(self.graph.graph.keys()))
        #self.current_state = start_node
        
        x_initial = 329.6474941596216
        y_initial =  660.1966888688361
        theta_initial = -0.20238621771937623
        velocity_initial = 5.108549775006556  
        return (x_initial, y_initial, velocity_initial,theta_initial)
   


    def compute_reward(self, action, is_destination):
        reward = 10

        # check if the episode is done
        return reward


    def compute_reward_disc(self, current_state, action):#, is_destination):
        """
        Computes the reward for transitioning from the current_state to next_state via action.

        Args:
            current_state: discretized current velocity, position and rotation action,
            action:
            next_state: discretized next velocity, position and rotation action
            is_destination: True is next_state is a final state, False otherwise.

        Return:
            float: final reward
        
        Ref: https://arxiv.org/pdf/2405.01440
        """
        #TODO: penalize progress away from the goal, reward progress toward the goal
        #TODO: rewards between -1,0,1
        #TODO: add objects
        # Initialize reward components
        speed_reward = 0
        safety_reward = 0
        smoothness_reward = 0
        progress_reward = 0

        #TODO: modify to handle discretized state, consider objects
        vel_predicate = current_state[1].value[0]
        velocity = Velocity[str(vel_predicate)[:-1].split('(')[1]]
        steer_predicate = current_state[2].value[0]
        steer_angle = Rotation[str(steer_predicate)[:-1].split('(')[1]]

        # Encourage maintaining a safe and moderate speed
        if velocity in [Velocity.LOW, Velocity.MEDIUM]:
            speed_reward += 0.2
        elif velocity in [Velocity.HIGH, Velocity.VERY_HIGH]:
            speed_reward -= 0.2 # Penalize very high speeds for safety reasons

        # TODO: Penalize stopping in potentially unsafe or unnecessary situations (when car is in the back for example)
        #if velocity == Velocity.STOPPED:
        #    safety_reward -= 0.1

        #if object_back and velocity.stopped --> penalize

        # Encourage smooth driving: penalize sudden actions that might indicate aggressive driving, or line change
        if action in [Action.TURN_LEFT, Action.TURN_RIGHT, Action.GAS_TURN_LEFT, Action.GAS_TURN_RIGHT, Action.BRAKE_TURN_LEFT, Action.BRAKE_TURN_RIGHT]:
            smoothness_reward -= 0.1
        elif action == Action.STRAIGHT:
            smoothness_reward += 0.2
        #Action.GAS, Action.BRAKE

        if steer_angle in [Rotation.LEFT, Rotation.RIGHT]:
            smoothness_reward -=0.1
        elif steer_angle in [Rotation.SLIGHT_LEFT, Rotation.SLIGHT_RIGHT]:
            smoothness_reward -=0.05
        else:
            smoothness_reward +=0.1

        # To encourage actions that lead to progress towards a goal, give positive reward if current state is a intermediate destination
        is_destination = True if action == None else False
        if is_destination:
            progress_reward += 1
        else:
            progress_reward -=0.1

        return speed_reward, safety_reward, smoothness_reward, progress_reward
    
    def step():
        pass

