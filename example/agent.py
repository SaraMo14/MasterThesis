
from example.discretizer.utils import Action
import numpy as np

from pgeon.agent import Agent

class SelfDrivingAgent(Agent):
    def __init__(self):
        super(SelfDrivingAgent, self).__init__()
        
        self.dt=0.5
        self.wheel_base = 2.588 #in meters. Ref: https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        
        #self.visited_states = set()
        #self.newly_discovered_states = set()



    def move(self, action: Action):
        """
        Function that given current state and action returns the next continuous state.
        
        Args:
            self --> current_state (tuple): (x, y, v, steering_angle) where steering_angle is in radians
            action: Action taken (e.g., "go straight", "turn", "gas", etc.)
        
        Returns:
        tuple: Updated state (x', y', v', new_steering_angle)


        ref: https://es.mathworks.com/help/mpc/ug/obstacle-avoidance-using-adaptive-model-predictive-control.html

        """
        #TODO: how will detected objects change their state?
        x, y, velocity, theta = self.current_state
       
        if action in (Action.TURN_LEFT, Action.GAS_TURN_LEFT, Action.BRAKE_TURN_LEFT):
            steer_angle = -30
        if action in (Action.TURN_RIGHT, Action.GAS_TURN_RIGHT,  Action.BRAKE_TURN_RIGHT):
            steer_angle = +30
        else:
            steer_angle = 0
        
        steer_angle_rad = np.deg2rad(steer_angle)

        #update velocity
        if action in (Action.GAS,  Action.GAS_TURN_LEFT,Action.GAS_TURN_RIGHT):
            velocity += 0.2 #self.discretizer.eps_vel ()
        elif action in (Action.BRAKE,  Action.BRAKE_TURN_LEFT, Action.BRAKE_TURN_RIGHT):
            velocity -= 0.2 #self.discretizer.eps_vel ()
        
        #update orientation (theta) based on steering angle and velocity
        new_theta = theta - (velocity / self.wheel_base) * np.tan(steer_angle_rad) * self.dt if action not in [Action.STRAIGHT, Action.GAS, Action.BRAKE] else theta

        # Update position based on velocity and orientation
        delta_x = velocity * np.cos(new_theta) * self.dt
        delta_y = velocity * np.sin(new_theta) * self.dt

        # Calculate new position
        new_x = x + delta_x
        new_y = y + delta_y
    
        return (new_x, new_y, velocity, new_theta)
    
    
    def update_state_tracking(self, state):
        """
        Updates the visited and newly discovered states based on the current state.
        """
        if state not in self.visited_states:
            self.newly_discovered_states.add(state)
        self.visited_states.add(state)


    def compute_proportion_of_discovery(self):
        """
        Computes the proportion of newly discovered states to total visited states.
        """
        if len(self.visited_states) == 0:
            return 0  # Prevent division by zero
        return len(self.newly_discovered_states) / len(self.visited_states)

    