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

  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class Rotation(Enum):
  LEFT = auto()
  SLIGHT_LEFT = auto()
  FORWARD = auto()
  SLIGHT_RIGHT = auto()
  RIGHT = auto()

  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class Position():
    def __init__(self, x,y):
        self.x = x
        self.y = y
        #self.z = 0

    def __str__(self) -> str:
        return f'Position({self.x}, {self.y})'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

class Action(Enum):
  IDLE = auto() 
  TURN_LEFT = auto()
  TURN_RIGHT = auto()
  GAS = auto() 
  BRAKE = auto()
  REVERSE = auto()
  STRAIGHT = auto() #car keep going straight at same pace
  GAS_TURN_RIGHT= auto()
  GAS_TURN_LEFT= auto()
  #GAS_STRAIGHT = auto()
  BRAKE_TURN_RIGHT = auto()  
  BRAKE_TURN_LEFT = auto()
  #BRAKE_STRAIGHT = auto()
  #TODO:differentiate between sharp and slight accelaraion, slight turn, ..., lane keeping, preparing to lane change, and lane changing (more of intentations)


class AVDiscretizer(Discretizer):
    def __init__(self):
        super(AVDiscretizer, self).__init__()

        self.unique_states: Dict[str, int] = {}
        self.velocity_thr = [0.2, 6, 11, 17, 22] #m/s while in km/h would be[0, 20, 40, 60, 80] 
        self.rotation_thr = [-2*np.pi/3, -np.pi/3, np.pi/3, 2*np.pi/3]  #[-2.5, -1, 0., 1, 2.5] #radiants
                             
        self.eps_rot = 0.2
        self.eps_vel = 0.5 #0.2
        self.eps_acc = 0.2
        self.eps_pos_x = 0.1
        self.eps_pos_y = 0.2
        #self.eps_heading_change = 0.02
        
        self.frequency = 0.5 #2Hz
    @staticmethod
    def is_close(a, b, eps=0.1):
        return abs(a - b) <= eps

    
    def discretize(self,
                   state: np.ndarray
                   ) -> Tuple[Predicate, Predicate, Predicate]:
        x, y, velocity, rotation = state 

        pos_predicate = self.discretize_position((x,y))
        mov_predicate = self.discretize_speed(self.velocity_thr, velocity)
        rot_predicate = self.discretize_rotation(self.rotation_thr, rotation)

        return (Predicate(Position, [pos_predicate.x, pos_predicate.y]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]))

    def discretize_position(self, position, chunk_size = 4):
        '''
        Discretizes the position of a point (x, y) into chunks of specified size. The position refers to the position of the LIDAR sensor on top of the vehicle, in the center. '''
        x,y = position

        x_chunk_index = int(np.floor(x / chunk_size)) #takes larger int <= (x/chunk_size)
        y_chunk_index = int(np.floor(y / chunk_size)) 
        #TODO: caso in cui è in un incrocio
        return Position(x_chunk_index,y_chunk_index) 

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

        state_to_be_discretized = ['x', 'y', 'velocity', 'yaw']
        state_columns_for_action = ['delta_local_x', 'delta_local_y', 'velocity', 'heading_change_rate', 'acceleration']
        n_states = len(states)

        previous_scene_token = states.iloc[0]['scene_token']
        for i in range(n_states-1):
            current_scene_token = states.iloc[i]['scene_token']

            # discretize current state
            current_state_to_discretize = states.iloc[i][state_to_be_discretized].tolist()
            discretized_current_state = self.discretize(current_state_to_discretize)
            current_state_str = self.state_to_str(discretized_current_state)
            current_state_id = self.add_unique_state(current_state_str)

            #check if is scene destination state
            if current_scene_token != previous_scene_token:
                previous_scene_token = current_scene_token
                action_id = None
            else:
                # Determine action based on the full state information
                current_state_for_action = states.iloc[i][state_columns_for_action].tolist()
                next_state_for_action = states.iloc[i+1][state_columns_for_action].tolist()
                action = self.determine_action(current_state_for_action, next_state_for_action)
                action_id = self.get_action_id(action)

                # Debugging print statements
                print(f'State {i}: {current_state_for_action}')
                print(f'Discretized state: {i} {discretized_current_state}')
                print(f'Action: {action}')
                print()
        
            trajectory.extend([current_state_id, action_id])        
        #add last state
        last_state_to_discretize = states.iloc[n_states-1][state_to_be_discretized].tolist()
        discretized_last_state = self.discretize(last_state_to_discretize)
        last_state_str = self.state_to_str(discretized_last_state)
        last_state_id = self.add_unique_state(last_state_str)
        trajectory.extend([last_state_id, None, None])        

        return trajectory




    def determine_action(self, current_state, next_state) -> Action:
        delta_x0, delta_y0, vel_t, rot_rate_t0, acc_t0 = current_state
        delta_x1, delta_y1, vel_t1, rot_rate_t1, acc_t1 = next_state

        if self.is_close(delta_x1, 0, self.eps_pos_x) and self.is_close(delta_y1, 0, self.eps_pos_y) and self.is_close(vel_t, 0, self.eps_vel):
            return Action.IDLE
        if delta_y1 < -self.eps_pos_y:
            return Action.REVERSE

        # determine acceleration
        if acc_t1 > self.eps_acc:
            acc_action = Action.GAS
        elif acc_t1 < -self.eps_acc:
            acc_action = Action.BRAKE
        else:
            acc_action = None

        # determine direction
        if delta_x1 > self.eps_pos_x:
            dir_action = Action.TURN_RIGHT
        elif delta_x1 < -self.eps_pos_x:
            dir_action = Action.TURN_LEFT
        else:
            dir_action = Action.STRAIGHT

        if acc_action == Action.GAS and dir_action == Action.TURN_RIGHT:
            return Action.GAS_TURN_RIGHT
        elif acc_action == Action.GAS and dir_action == Action.TURN_LEFT:
            return Action.GAS_TURN_LEFT
        elif acc_action == Action.GAS and dir_action == Action.STRAIGHT:
            return Action.GAS#_STRAIGHT
        elif acc_action == Action.BRAKE and dir_action == Action.TURN_RIGHT:
            return Action.BRAKE_TURN_RIGHT
        elif acc_action == Action.BRAKE and dir_action == Action.TURN_LEFT:
            return Action.BRAKE_TURN_LEFT
        elif acc_action == Action.BRAKE and dir_action == Action.STRAIGHT:
            return Action.BRAKE#_STRAIGHT
        elif acc_action is None:
            # fallback to direction if no acceleration action was determined
            return dir_action

        # if no other conditions met
        return Action.IDLE

    def add_unique_state(self, state_str: str) -> int:
        if state_str not in self.unique_states:
            self.unique_states[state_str] = len(self.unique_states)
        return self.unique_states[state_str]

    @staticmethod
    def get_action_id(action):
            return action.value
    
    @staticmethod
    def get_action_from_id(action_id):
        for action in Action:
            if action.value == action_id:
                return action
        raise ValueError("Invalid action ID")


    def state_to_str(self,
                     state: Tuple[Predicate, Predicate, Predicate]
                     ) -> str:

        return '&'.join(str(pred) for pred in state)

    def str_to_state(self, state_str: str) -> Tuple[Predicate, Predicate, Predicate]:
        pos_str, vel_str, rot_str = state_str.split('&')
        
        x, y = map(int, pos_str[len("Position("):-1].split(','))
        
        mov_predicate = Velocity[vel_str[:-1].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-1].split('(')[1]]
        
        return (Predicate(Position, [x, y]),
                        Predicate(Velocity, [mov_predicate]),
                        Predicate(Rotation, [rot_predicate]))
    
    
    def nearest_state(self, state, chunk_size = 4):
        #TODO: put chunk size as varible of the discretizer?
        og_position, og_velocity, og_angle = state
        
        x,y = og_position.value
        # Generate nearby positions considering discretization
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x,new_y = x + dx * chunk_size, y + dy * chunk_size
                if (new_x, new_y) != (x, y):
                    yield Predicate(Position, [new_x, new_y]), og_velocity, og_angle

        for v in Velocity:
            if v != og_velocity.value:
                yield og_position, Predicate(Velocity, v), og_angle
        for r in Rotation:
            if r != og_angle.value:
                yield og_position, og_velocity, Predicate(Rotation, r)

        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                new_x,new_y = x + dx * chunk_size, y + dy * chunk_size

                if (new_x, new_y) != (x, y):
                    for v in Velocity:
                        if v != og_velocity.value:
                            yield Predicate(Position, [new_x, new_y]), Predicate(Velocity, v), og_angle
                    for r in Rotation:
                        if r != og_angle.value:
                            yield Predicate(Position, [new_x, new_y]), og_velocity, Predicate(Rotation, r)


    def all_actions(self):
        return list(Action) 
        
    #TODO: update
    def get_predicate_space(self):
        all_tuples = []

        #for p in Position:
            #for v in Velocity:
                #for r in Rotation:
                    #all_tuples.append((p,v,r))
        for v in Velocity:
            for r in Rotation:
                all_tuples.append((v,r))
        return all_tuples

