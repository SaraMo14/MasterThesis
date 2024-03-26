from enum import Enum, auto

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pgeon.discretizer import Discretizer, Predicate


class Velocity(Enum):
  STOPPED = auto()
  LOW = auto()
  MEDIUM = auto()
  HIGH = auto()
  VERY_HIGH = auto()

class Rotation(Enum):
  LEFT = auto()
  #SLIGHT_LEFT = auto()
  STRAIGHT = auto()
  #SLIGHT_RIGHT = auto()
  RIGHT = auto()

class RotationRate(Enum):
  RIGHT_TURN = auto()
  SLIGHT_RIGHT = auto()
  STRAIGHT = auto()
  SLIGHT_LEFT = auto()
  LEFT_TURN = auto()

class Position(): #output of LIDAR
  def __init__(self):
    self.x = 0
    self.y = 0
    self.z = 0
  #LEFT = auto()
  #CENTER = auto()
  #RIGHT = auto()

#class DeltaX(Enum): #describes lateral position variation of the car within the lane
  #LEFT = auto() <0
  #CENTER = auto()
  #RIGHT = auto() >0

#class DeltaY(Enum): #represents the longitudinal position (front-to-back)  variation of the car 
  #FORWARD = auto() <0
  #STOPPED = auto()
  #BACKWARD = auto() >0

#Action Space: Discrete space that  consists of Lateral movement and Longitudinal movement.

class Action(Enum):
  IDLE = auto() #dist(current(x,y),next(x,y)) < eps very small
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

        self.unique_states = set()
        #TODO: modify values
        self.velocity_thr = [0, 4, 8, 14, 22] #m/s while in km/h would be[0, 15, 30, 50, 80] 
        self.rotation_thr = [-1, 0., 1] #degrees

    @staticmethod
    def is_close(a, b, eps=0.1):
        return abs(a - b) <= eps

    
    def discretize_state(self,
                   state: np.ndarray
                   ) -> Tuple[Predicate, Predicate, Predicate]: #frequence: 2Hz (2 frame per second/ 1 frame per 0.5 second)
        position, velocity, rotation, _, _= state 

        pos_predicate = self.discretize_position(position)
        mov_predicate = self.discretize_speed(self.velocity_thr, velocity)
        rot_predicate = self.discretize_rotation(self.rotation_thr, rotation)
        #self.angular_velocity = self.discretize_rotation_rate(angular_velocity)

        return (Predicate(Position, [pos_predicate]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]))

    def discretize_position(self, position, chunk_size = 4):
        '''
        Discretizes the position of a point (x, y) into chunks of specified size.
        The position refers to the position of the LIDAR sensor on top of the vehicle, in the center.

        Args:
            position: The x,y and z-coordinate of the point.
            chunk_size (float, optional): The size of each chunk in meters. Defaults to 4.

        Returns:
            Tuple[int, int]: The indices of the closest chunk in the x and y directions.

            
        '''
        #Possible improvement: discretize based on lane-centric representation (Ref:https://hal.science/hal-01908175/document)

        
        x,y,_= position[0], position[1], position[2] #position
        
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
        
        return x_chunk_index, y_chunk_index
                

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


    def compute_trajectory(self, states):#, filenames):
        """
            Discretizes a trajectory (list of states) and stores unique states.

            Args:
            trajectory: A DataFrame or np.arrays (to be decided) of states of instance of scene.

            Returns:
            A list of identifiers for the discretized states.
        """
        trajectory = []

        for i in range(len(states)-1):
            # Define the columns that constitute the state information
            state_columns = ['translation', 'velocity', 'yaw', 'heading_change_rate', 'acceleration']

            # Extract the current and next states using .iloc for row indexing and the list of columns for column indexing
            current_state = states.iloc[i][state_columns].tolist()
            next_state = states.iloc[i+1][state_columns].tolist()
            discretized_current_state = self.discretize_state(current_state)
            action = self.determine_action(current_state, next_state) 

            # Convert current state to a string for tracking unique states
            current_state_str = self.state_to_str(discretized_current_state)
            
            print(f' current state: {current_state}')
            print(f' next state: {next_state}')
            
            print(f'discretized current state: {current_state_str}')
            print(f'action: {action}')            

            ######### display image for visual test #############
            #nuscenes.render_instance('0c3833f73f0e43288758cb87c9d2a4fe')
            #image_path = filenames['filename'].iloc[i]  # Gets the first image filename

            #img = mpimg.imread('/home/saramontese/Desktop/MasterThesis/example/nuscenes/data/sets/nuscenes/' + image_path)  # Reads the image from the path
            #plt.imshow(img)  # Displays the image
            #plt.axis('off')  # Hides the axis
            #plt.show()

            ######################################################



            if current_state_str not in self.unique_states:
                self.unique_states.add(current_state_str)
            
            # Get identifiers for the current state and action
            current_state_id = self.get_state_identifier()

            #TODO: update action_id = action.value#Action.index(action)
        
            #trajectory.append((current_state_id, action_id))
            trajectory.append(current_state_id)
            #TODO: update how to append actions trajectory.append(action_id)

        return trajectory
    

    #TODO: change location and definition of this function
    #def retrieve_sample_image(self, df, instance_token):
        '''
            Args:
            df: DataFrame of instances
            instance_token: Instance that needs to be detected or tracked by an AV

            Return:
            Lidar and front camera images of the AV
        '''
        #nuscenes = 
        #return self.nuscenes.render_sample(instance_token)


    
    def determine_action(self, current_state, next_state, eps=None) -> int:
        
        '''
            Given full state(t) and state(t+1), returns the inferred action.

            Args:
            current_state: undiscretized state(t)
            next_state: undiscretized state(t+1)

            Return:
            ID numbers of actions performed
        '''
        pos_t, vel_t, _, _, _ = current_state
        pos_t1, vel_t1, _, rot_rate_t1, acc_t1 = next_state
        
        x_t, y_t, _= pos_t[0], pos_t[1], pos_t[2]
        x_t1, y_t1, _= pos_t1[0], pos_t1[1], pos_t1[2]
        
        # Calculate differences
        dt = 0.5
        x_diff = (x_t1 - x_t)/dt
        y_diff = (y_t1 - y_t)/dt
        #z_diff = (z_t1 - z_t)/dt
        vel_diff = (vel_t1 - vel_t)/dt
        #rot_diff = rot_t1 - rot_t

        #set epsilon (to be done in __init__?)
        #eps_rot = 0.02
        eps_vel = 0.2
        eps_pos = 0.2
        #eps_acc = 2

        actions = set()
        
        # Check for IDLE condition first
        if self.is_close(x_diff, 0, eps_pos) and self.is_close(y_diff, 0, eps_pos) and self.is_close(vel_diff, 0, eps_vel):
            actions.add(Action.IDLE)
        # Movement and rotation logic
        elif y_diff < - eps_pos:
            if vel_diff > eps_vel: # and self.is_close(rot_diff, 0, eps_rot):
                actions.add(Action.GAS)
            elif vel_diff < -eps_vel: # and self.is_close(rot_diff, 0, eps_rot):
                actions.add(Action.BRAKE)
            #else: 
            #    actions.add(Action.STRAIGHT) #goes ahead at same pace
            if rot_rate_t1 > 0 :#eps_rot:
                actions.add(Action.TURN_RIGHT)
            elif rot_rate_t1 < 0: #-eps_rot:
                actions.add(Action.TURN_LEFT)
            #elif  rot_rate_t1 == 0 and self.is_close(vel_diff, 0, eps_vel):
                
        elif y_diff > eps_pos: 
            actions.add(Action.REVERSE)

        # Fallback to IDLE if no other actions are determined, ensuring it's not added if other actions exist
        if not actions:
            actions.add(Action.IDLE)

        return actions

        '''
        if self.is_close(x_diff, 0, eps_pos) and self.is_close(y_diff, 0, eps_pos) and self.is_close(vel_diff, 0, eps_vel):
            return Action.IDLE
        elif vel_diff > eps_vel and self.is_close(rot_diff, 0, eps_rot) and y_diff < eps_pos: 
            return Action.GAS
        elif vel_diff < -eps_vel and self.is_close(rot_diff, 0, eps_rot) and y_diff < eps_pos:
            return Action.BRAKE
        elif rot_diff > eps_rot:
            if vel_diff > eps_vel:
                return Action.GAS_AND_TURN_RIGHT
            elif vel_diff < -eps_vel:
                return Action.BRAKE_AND_TURN_RIGHT
            else:
                return Action.TURN_RIGHT
        elif rot_diff < -eps_rot:
            if vel_diff > eps_vel:
                return Action.GAS_AND_TURN_LEFT
            elif vel_diff < -eps_vel:
                return Action.BRAKE_AND_TURN_LEFT
            else:
                return Action.TURN_LEFT
        elif y_diff > eps_pos:
            return Action.REVERSE
        elif x_diff > eps and self.is_close(rot_diff, 0, eps_rot) and self.is_close(vel_diff,0, eps_vel): #TODO: fix position interpretation
            return Action.STRAIGHT
        else:
            return Action.IDLE
        
        #TODO: add Lane change inference 
        
        '''


    def get_state_identifier(self):
        """
        Generates a unique identifier for a state

        """
        state_id = len(self.unique_states)+1
        return state_id


    def state_to_str(self,
                     state: Tuple[Predicate, Predicate, Predicate]
                     ) -> str:

        return '&'.join(str(pred.value) for pred in state)
        #return '&'.join(str(pred) for pred in state)

    
    #TODO: update
    def str_to_state(self, state: str):
        pos, vel, rot = state.split('&')
        pos_predicate = Position[pos[:-1].split('(')[1]]
        mov_predicate = Velocity[vel[:-1].split('(')[1]]
        rot_predicate = Rotation[rot[:-1].split('(')[1]]

        return (Predicate(Position, [pos_predicate]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]))


    def transition_matrix(states_actions_list, num_states, num_actions):
        
        """
        Build transition matrices from state-action pairs.

        Args:
        - states_actions_list: A list in the format [state0, action0, state1, action1, ...]
        - num_states: The total number of unique states.
        - num_actions: The total number of unique actions.

        Returns:
        A list of transition matrices, one for each action.
        """

        # initialize transition matrices for each action
        transition_matrices = [np.zeros((num_states, num_states)) for _ in range(num_actions)]

        # populate the transition matrices
        for i in range(0, len(states_actions_list) - 2, 2):  # Step by 2 to move from state-action pair to the next pair
            state_id = states_actions_list[i]
            action_id = states_actions_list[i+1]
            next_state_id = states_actions_list[i+2]

            # Assuming action_id is the index for the action matrix and is 0-indexed
            transition_matrices[action_id][state_id, next_state_id] += 1

        # convert counts to probabilities
        for action_id in range(num_actions):
            counts = transition_matrices[action_id]
            row_sums = counts.sum(axis=1, keepdims=True)
            # Avoid division by zero for states with no outgoing transitions
            transition_matrices[action_id] = np.divide(counts, row_sums, where=row_sums!=0)

        return transition_matrices



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

