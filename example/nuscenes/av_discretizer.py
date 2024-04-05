from enum import Enum, auto

import numpy as np
from typing import Tuple
from typing import Dict

from pgeon.discretizer import Discretizer, Predicate


class Velocity(Enum):
  STOPPED = auto()
  LOW = auto()
  MEDIUM = auto()
  HIGH = auto()
  VERY_HIGH = auto()


class Rotation(Enum):
  LEFT = auto()
  SLIGHT_LEFT = auto()
  FORWARD = auto()
  SLIGHT_RIGHT = auto()
  RIGHT = auto()


class Position():
    def __init__(self, x,y):
        self.x = x
        self.y = y
        #self.z = 0
        #LEFT = auto()
        #CENTER = auto()
        #RIGHT = auto()

    def __str__(self) -> str:
        return f'Position({self.x}, {self.y})'


class Action(Enum):
  IDLE = auto() 
  TURN_LEFT = auto()
  TURN_RIGHT = auto()
  GAS = auto() #car keeps going straight, but faster
  BRAKE = auto() #car keeps going straight, but slower
  GAS_AND_TURN_LEFT = auto()
  GAS_AND_TURN_RIGHT = auto()
  BRAKE_AND_TURN_LEFT = auto()
  BRAKE_AND_TURN_RIGHT = auto()
  REVERSE = auto()
  STRAIGHT = auto() #car keep going straight at same pace

  #TODO:
  #differentiate between sharp and slight accelaraion, turn ..
  #lane keeping, preparing to lane change, and lane changing (more of intentations)


class AVDiscretizer():
    def __init__(self):
        super(AVDiscretizer, self).__init__()

        self.unique_states: Dict[str, int] = {}
        self.unique_actions=  {
            (Action.IDLE,): 0,
            (Action.GAS,): 1,
            (Action.BRAKE,): 2,
            (Action.TURN_RIGHT,): 3,
            (Action.TURN_LEFT,): 4,
            (Action.STRAIGHT,): 5,
            (Action.REVERSE,): 6,
            #combinations:
            (Action.GAS, Action.TURN_RIGHT): 7,
            (Action.GAS, Action.TURN_LEFT): 8,
            (Action.GAS, Action.STRAIGHT): 9,
            (Action.BRAKE, Action.TURN_RIGHT): 10,
            (Action.BRAKE, Action.TURN_RIGHT): 11,
            (Action.BRAKE, Action.STRAIGHT): 12,
            # Add other combinations as needed
        }
        
        self.velocity_thr = [0.2, 6, 11, 17, 22] #m/s while in km/h would be[0, 20, 40, 60, 80] 
        self.rotation_thr = [-1*np.pi/3, -np.pi/3, np.pi/3, 2*np.pi/3]  #[-2.5, -1, 0., 1, 2.5] #radiants

        self.eps_rot = 0.2
        self.eps_vel = 0.5 #0.2
        self.eps_pos_x = 0.1
        self.eps_pos_y = 0.2
        
    @staticmethod
    def is_close(a, b, eps=0.1):
        return abs(a - b) <= eps

    
    def discretize_state(self,
                   state: np.ndarray
                   ) -> Tuple[Predicate, Predicate, Predicate]:
        x, y, _, velocity, rotation = state 

        pos_predicate = self.discretize_position((x,y))
        mov_predicate = self.discretize_speed(self.velocity_thr, velocity)
        rot_predicate = self.discretize_rotation(self.rotation_thr, rotation)

        return (Predicate(Position, [(pos_predicate.x, pos_predicate.y)]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]))

    def discretize_position(self, position, chunk_size = 4):
        '''
        Discretizes the position of a point (x, y) into chunks of specified size.
        The position refers to the position of the LIDAR sensor on top of the vehicle, in the center.

        Args:
            x,y,z: The x,y and z-coordinate of the point.
            chunk_size (float, optional): The size of each chunk in meters. Defaults to 4.

        Returns:
            Tuple[int, int]: The indices of the closest chunk in the x and y directions.

            
        '''
        #Possible improvement: discretize based on lane-centric representation (Ref:https://hal.science/hal-01908175/document)
        x,y = position

        x_chunk_index = int(np.floor(x / chunk_size)) #takes larger int <= (x/chunk_size)
        y_chunk_index = int(np.floor(y / chunk_size)) 
        #we use np.floor to ensure that the index represents the chunk to the left of the point

        #calculate midpoint of the chunck
        x_midpoint = (x_chunk_index+0.5) * chunk_size
        y_midpoint = (y_chunk_index+0.5) * chunk_size
        
        #calculate Euclidean distance from coordinates to chunk midpoint
        distance = np.sqrt((x-x_midpoint)**2+ (y-y_midpoint)**2)

        #if distance is greater than half the chunk size, assign to the closest chunk
        if distance > chunk_size / 2:
            if x < x_midpoint: 
                x_chunk_index-=1
            if y < y_midpoint: 
                y_chunk_index-=1
        #elif distance == chunk_size / 2:   
        #TODO: caso in cui è in un incrocio
        position = Position(x_chunk_index,y_chunk_index)
        #return x_chunk_index, y_chunk_index
        return position 

    @staticmethod
    def discretize_speed(thresholds, speed) -> Velocity:
        for i, threshold in enumerate(thresholds):
            if speed <= threshold: 
                return Velocity(i + 1)
        return Velocity.VERY_HIGH

    
    @staticmethod
    def discretize_rotation(thresholds, rotation) -> Rotation:
        for i, threshold in enumerate(thresholds):
            if rotation <= threshold:  
                return Rotation(i + 1)
        return Rotation.RIGHT  
    
    
    #def discretize_acceleration(self, acceleration):
      #Values Reference: https://pure.tue.nl/ws/portalfiles/portal/50922576/matecconf_aigev2017_01005.pdf


    def compute_trajectory(self, states):
        """
            Discretizes a trajectory (list of states) and stores unique states and actions.

            Args:
                states: DataFrame containing state information for each time step.

            Returns:
                List containing tuples of (current state ID, action ID, next state ID).
        """
        trajectory = []

        state_to_be_discretized = ['x', 'y', 'z', 'velocity', 'yaw']
        state_columns_for_action = ['delta_local_x', 'delta_local_y', 'velocity', 'yaw', 'heading_change_rate', 'acceleration', 'timestamp']
        n_states = len(states)

        for i in range(n_states-1):
            # discretize current state
            current_state_to_discretize = states.iloc[i][state_to_be_discretized].tolist()
            discretized_current_state = self.discretize_state(current_state_to_discretize)
            current_state_str = self.state_to_str(discretized_current_state)
            current_state_id = self.add_unique_state(current_state_str)

            # Determine action based on the full state information
            current_state_for_action = states.iloc[i][state_columns_for_action].tolist()
            next_state_for_action = states.iloc[i+1][state_columns_for_action].tolist()
            action = self.determine_action(current_state_for_action, next_state_for_action)
            action_id = self.get_action_id(action)

            # Debugging print statements
            print(f'State {i}: {current_state_for_action}')
            print(f'Discretized state: {i} {current_state_str}')
            print(f'Action: {action}')
            print()
        
            trajectory.extend([current_state_id, action_id])        
        #add last state
        last_state_to_discretize = states.iloc[n_states-1][state_to_be_discretized].tolist()
        discretized_last_state = self.discretize_state(last_state_to_discretize)
        last_state_str = self.state_to_str(discretized_last_state)
        last_state_id = self.add_unique_state(last_state_str)
        trajectory.append(last_state_id)
        
        return trajectory


    def determine_action(self, current_state, next_state) -> int:
        
        '''
            Given full state(t) and state(t+1), returns the inferred action.

            Args:
            current_state: undiscretized state(t)
            next_state: undiscretized state(t+1)

            Return:
            ID numbers of actions performed
        '''
        delta_x0, delta_y0, vel_t, yaw, rot_rate_t0, acc_t0, t0 = current_state
        delta_x1, delta_y1, vel_t1, yaw_t1, rot_rate_t1, acc_t1, t1 = next_state
        
        
        # Calculate differences
        #dt = (t1-t0).total_seconds()
        #rot_diff = (rot_rate_t1 - rot_rate_t0)/dt

        actions = set()
        
        # Check for IDLE condition first
        if self.is_close(delta_x1, 0, self.eps_pos_x) and self.is_close(delta_y1, 0, self.eps_pos_y) and self.is_close(vel_t, 0, self.eps_vel): #TODO: should i say more about velocity?
            actions.add(Action.IDLE)

        elif delta_y1 > self.eps_pos_y:
            if acc_t1 > self.eps_vel: 
                actions.add(Action.GAS)
            elif acc_t1 < -self.eps_vel: 
                actions.add(Action.BRAKE)

            if delta_x1 > self.eps_pos_x:#rot_diff > eps_rot: #or rot_diff?
                actions.add(Action.TURN_RIGHT)
            elif delta_x1< -self.eps_pos_x:#rot_diff < -eps_rot: #or rot_diff?
                actions.add(Action.TURN_LEFT)
            else:
                actions.add(Action.STRAIGHT)#elif  rot_rate_t1 == 0 and self.is_close(vel_diff, 0, eps_vel):
                
        elif delta_y1 < -self.eps_pos_y: 
            actions.add(Action.REVERSE)

        if not actions:
            actions.add(Action.IDLE)

        return actions
    

    def add_unique_state(self, state_str: str) -> int:
        """
        Adds a new unique state to the unique_states dictionary if it doesn't already exist, assigning it the next available index.

        Args:
            state_str (str): The state string to be added or found in the unique states.

        Returns:
            int: The index (ID) of the state in the unique_states dictionary.
        """
        if state_str not in self.unique_states:
            self.unique_states[state_str] = len(self.unique_states)
        return self.unique_states[state_str]

    

    def get_action_id(self, actions):
        """
            Retrieve the unique code for a given set of actions.
            
            Args:
            actions (set of Action): The set of actions for which to retrieve the code.
            
            Returns:
            int: The unique code corresponding to the set of actions, or -1 if not found.
        """
        # Convert the set to a sorted tuple since sets are unordered and cannot be used as dictionary keys directly
        actions_tuple = tuple(sorted(actions, key=lambda action: action.value))
        
        return self.unique_actions.get(actions_tuple, -1)
    

    
    def state_to_str(self,
                     state: Tuple[Predicate, Predicate, Predicate]
                     ) -> str:
        return '&'.join(str(pred.value) for pred in state)
    
    '''
    def str_to_state(self, state_str: str) -> Tuple[Predicate, Predicate, Predicate]:
        pos_str, vel_str, rot_str = state_str.split('&')
        
        # Assuming pos_str format is "Position(x,y,z)"
        # Extracting numeric values for x, y, and z from the position string
        pos_vals = pos_str.split('(')[1].rstrip(')').split(',')
        x, y = map(float, pos_vals)  # Convert strings to floats
        
        # Create a Position object with x, y
        pos_predicate = Position()
        pos_predicate.x, pos_predicate.y = x, y

        # Assuming vel and rot are still using Enum names directly
        mov_predicate = Velocity[vel_str]
        rot_predicate = Rotation[rot_str]

        return (Predicate(Position, [pos_predicate]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]))

    '''
    def str_to_state(self, state_str: str) -> Tuple[Position, Velocity, Rotation]:
        # Split the state string into its components
        pos_str, vel_str, rot_str = state_str.split('&')
        
        # Extract the position values (assuming format is [(x, y)])
        pos_vals = pos_str.strip("[]()").split(',')
        x, y = map(int, pos_vals)  # Convert strings to integers
        
        # Create a Position object with x, y (assuming z is not provided and defaults to 0)
        pos = Position(x, y)
        
        # Extract the Velocity value (assuming format is [<Velocity.X: Y>])
        # We strip out the non-numeric characters and extract the enum name
        vel_enum_name = vel_str.split('.')[1].split(':')[0]
        vel = Velocity[vel_enum_name]

        # Extract the Rotation value (assuming format is [<Rotation.X: Y>])
        # Similar extraction process as for Velocity
        rot_enum_name = rot_str.split('.')[1].split(':')[0]
        rot = Rotation[rot_enum_name]

        return (pos.x, pos.y), vel, rot


    #TODO: update
    def nearest_state(self, state):
        '''
        Find states closest (in terms of discretized values) to a given input state.
        
        1.Iterate Over Single Variable Changes:

        It loops through all possible values in Position (LEFT, MIDDLE, RIGHT). If the current position value doesn't match the value being iterated upon, it creates a new state with the different position value but keeps the original velocity and angle values. This creates states that differ only in position from the input state.
        Similarly, it performs the same process for Velocity and Angle to find states that differ only in velocity or angle, respectively.
        
        
        2.Iterate Over Combinations of Two Variable Changes:

        After checking single variable changes, the function iterates through all possible combinations of two variables (position, velocity, and angle). For each combination, it checks if both values in the original state match the current combination.
        If there's a mismatch in only one variable, it creates a new state with the different value from the current combination while keeping the other two variables the same as the original state.
        '''
        
        og_position, og_velocity, og_angle = state
        
        for e in Position:
            if [e] != og_position.value:
                yield Predicate(Position, [e]), og_velocity, og_angle
        for e in Velocity:
            if [e] != og_velocity.value:
                yield og_position, Predicate(Velocity, [e]), og_angle
        for e in Rotation:
            if [e] != og_angle.value:
                yield og_position, og_velocity, Predicate(Rotation, [e])

        for e in Position:
            for f in Velocity:
                for g in Rotation:
                    amount_of_equals_to_og = \
                        int([e] == og_position.value) + int([f] == og_velocity.value) + int([g] == og_angle.value)
                    if amount_of_equals_to_og <= 1:
                        yield Predicate(Position, [e]), Predicate(Velocity, [f]), Predicate(Angle, [g])



    def all_actions(self):
        return list(Action) 

    def get_predicate_space(self):
        all_tuples = []

        for p in Position:
            for v in Velocity:
                for r in Rotation:
                    all_tuples.append((p,v,r))
        return all_tuples

