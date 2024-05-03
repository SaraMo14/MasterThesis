from enum import Enum, auto
from example.discretizer.utils import Action, Position, Velocity, Rotation, DetectedObject
import numpy as np
from typing import Tuple, Union, Dict
from pgeon.discretizer import Discretizer, Predicate

# this class is a discretizer for discretizers D0 (no cameras), D1A (generic and with 2 cameras)

class AVDiscretizer(Discretizer):
    def __init__(self):
        super(AVDiscretizer, self).__init__()

        self.unique_states: Dict[str, int] = {}
        self.state_to_be_discretized = ['x', 'y', 'velocity', 'steering_angle'] #yaw not needed 
        self.state_columns_for_action = ['delta_local_x', 'delta_local_y', 'velocity', 'acceleration', 'steering_angle'] #heading_change_rate not needed
     
        self.velocity_thr = [0.2, 6, 11, 17]#m/s while in km/h would be[0, 20, 40, 60] 
        #self.yaw_thr = [-2*np.pi/3, -np.pi/3, np.pi/3, 2*np.pi/3]  #[-2.5, -1, 0., 1, 2.5] #radiants
        self.rotation_thr = [-3, -0.3, 0.3 , 3]
        self.chunk_size = 4
        self.eps_rot = 0.4
        self.eps_vel = 0.5 #0.2
        self.eps_acc = 0.2
        self.eps_pos_x = 0.1
        self.eps_pos_y = 0.2
        
        self.frequency = 0.5 #2Hz
        
    @staticmethod
    def is_close(a, b, eps=0.1):
        return abs(a - b) <= eps

    
    def discretize(self,
                   state: np.ndarray, detections=None
                   ) -> Tuple[Predicate, Predicate, Predicate]:
        x, y, velocity, rotation = state 
        pos_predicate = self.discretize_position((x,y))
        mov_predicate = self.discretize_speed(velocity)
        rot_predicate = self.discretize_steering_angle(rotation)
        if detections is not None:
            detected_predicates = self.discretize_detections(detections)
            return (Predicate(Position, [pos_predicate.x, pos_predicate.y]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]),
                *detected_predicates)
        else:
            return (Predicate(Position, [pos_predicate.x, pos_predicate.y]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]))
        

    def discretize_detections(self, detections):
        detected_predicates = []
        for cam_type, objects in detections.items():
            obj = DetectedObject() if objects=="{}" else DetectedObject(cam_type) 
            
            detected_predicates.append(Predicate(DetectedObject, [obj]))
        return detected_predicates


    def discretize_steering_angle(self, steering_angle: float):
        for i, threshold in enumerate(self.rotation_thr):
            if steering_angle <= threshold:  
                return Rotation(i + 1)
        return Rotation.LEFT  


    def discretize_position(self, position):
        '''
        Discretizes the position of a point (x, y) into chunks of specified size. The position refers to the position of the LIDAR sensor on top of the vehicle, in the center. '''
        x,y = position

        x_chunk_index = int(np.floor(x / self.chunk_size)) #takes larger int <= (x/chunk_size)
        y_chunk_index = int(np.floor(y / self.chunk_size)) 
        #TODO: caso in cui è in un incrocio
        return Position(x_chunk_index,y_chunk_index) 

    def discretize_speed(self, speed) -> Velocity:
        for i, threshold in enumerate(self.velocity_thr):
            if speed <= threshold: 
                return Velocity(i + 1)
        return Velocity.VERY_HIGH

    

    def compute_trajectory(self, states):
        """
            Discretizes a trajectory (list of states) and stores unique states and actions.

            Args:
                states: DataFrame containing state information for each time step.

            Returns:
                List containing tuples of (current state ID, action ID, next state ID).
        """
        trajectory = []

        self.detection_cameras = [col for col in states.columns if 'CAM' in col] #NOTE: this should be assigned even when testing the PG (not only when computing the trajectory)
        
        n_states = len(states)

        for i in range(n_states-1):
            
            # discretize current state
            current_state_to_discretize = states.iloc[i][self.state_to_be_discretized].tolist()
            current_detection_info = states.iloc[i][self.detection_cameras] if len(self.detection_cameras)>0 else None
            discretized_current_state = self.discretize(current_state_to_discretize, current_detection_info)
            current_state_str = self.state_to_str(discretized_current_state)
            current_state_id = self.add_unique_state(current_state_str)
            #check if is scene destination state

            current_scene = states.iloc[i]['scene_token']
            next_scene = states.iloc[i+1]['scene_token']
            if current_scene != next_scene:
                action_id = None
            else:
                # Determine action based on the full state information
                current_state_for_action = states.iloc[i][self.state_columns_for_action].tolist()
                next_state_for_action = states.iloc[i+1][self.state_columns_for_action].tolist()
                action = self.determine_action(current_state_for_action, next_state_for_action)
                action_id = self.get_action_id(action)

                # Debugging print statements
                #print(f'State {i}: {current_state_for_action}')
                #print(f'Discretized state: {i} {discretized_current_state}')
                #print(f'Action: {action}')
                #print()
        
            trajectory.extend([current_state_id, action_id])        
        #add last state
        last_state_to_discretize = states.iloc[n_states-1][self.state_to_be_discretized].tolist()
        last_state_detections = states.iloc[n_states-1][self.detection_cameras] if len(self.detection_cameras)>0 else None
        discretized_last_state = self.discretize(last_state_to_discretize, last_state_detections)
        last_state_str = self.state_to_str(discretized_last_state)
        last_state_id = self.add_unique_state(last_state_str)
        trajectory.extend([last_state_id, None, None])        
        
        return trajectory




    def determine_action(self, current_state, next_state) -> Action:
        delta_x0, delta_y0, vel_t,  acc_t0, steer_t0 = current_state
        delta_x1, delta_y1, vel_t1,  acc_t1, steer_t1 = next_state

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
        if steer_t0 < -self.eps_rot:#delta_x1 > self.eps_pos_x:
            dir_action = Action.TURN_RIGHT
        elif steer_t0 > self.eps_rot:#delta_x1 < -self.eps_pos_x:
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
                     state: Tuple[Union[Predicate, ]]
                     ) -> str:
        return '&'.join(str(pred) for pred in state)

    
    def str_to_state(self, state_str: str) -> Tuple[Union[Predicate, ]]:
        split_str = state_str.split('&')
        pos_str, vel_str, rot_str = split_str[0:3]
        x, y = map(int, pos_str[len("Position("):-1].split(','))
        
        mov_predicate = Velocity[vel_str[:-1].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-1].split('(')[1]]

        if split_str[4:]:
            detected_predicates = []
            for cam_detections in split_str[3:]:
                cam_type = cam_detections[:-1].split('(')[1]#map(str.strip, cam_detections[:-2].split(','))
                detected_predicates.append(Predicate(DetectedObject, [DetectedObject(cam_type)]))
            return (Predicate(Position, [x, y]),
                        Predicate(Velocity, [mov_predicate]),
                        Predicate(Rotation, [rot_predicate]), *detected_predicates)
        else: 
            return (Predicate(Position, [x, y]),
                        Predicate(Velocity, [mov_predicate]),
                        Predicate(Rotation, [rot_predicate]))
    
    #TODO: review
    def nearest_state(self, state):
        og_position, og_velocity, og_angle, *detections = state

        x,y = og_position.value

        # Generate nearby positions considering discretization
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                new_x,new_y = x + dx, y + dy
                if (new_x, new_y) != (x, y):
                    yield Predicate(Position, [new_x, new_y]), og_velocity, og_angle, *detections

        for v in Velocity:
            if v != og_velocity.value:
                yield og_position, Predicate(Velocity, v), og_angle, *detections
        for r in Rotation:
            if r != og_angle.value:
                yield og_position, og_velocity, Predicate(Rotation, r), *detections
        
        if len(detections)>0:
            for objs in [(DetectedObject('CAM_FRONT'), DetectedObject()),
                (DetectedObject('CAM_FRONT'), DetectedObject('CAM_BACK')),
                (DetectedObject(), DetectedObject('CAM_BACK'))]:
                if objs not in detections:
                    yield og_position, og_velocity, og_angle, Predicate(DetectedObject, [objs[0]]), Predicate(DetectedObject, [objs[1]])
            
        

        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                new_x,new_y = x + dx, y + dy
                if (new_x, new_y) != (x, y):
                    for v in Velocity:
                        if v != og_velocity.value:
                            yield Predicate(Position, [new_x, new_y]), Predicate(Velocity, v), og_angle, *detections
                    for r in Rotation:
                        if r != og_angle.value:
                            yield Predicate(Position, [new_x, new_y]), og_velocity, Predicate(Rotation, r), *detections

                    if len(detections)>0:
                        for objs in [(DetectedObject('CAM_FRONT'), DetectedObject()),
                            (DetectedObject('CAM_FRONT'), DetectedObject('CAM_BACK')),
                            (DetectedObject(), DetectedObject('CAM_BACK'))]:
                            if objs not in detections:
                                yield Predicate(Position, [new_x, new_y]), og_velocity, og_angle, Predicate(DetectedObject, [objs[0]]), Predicate(DetectedObject, [objs[1]])



    def all_actions(self):
        return list(Action) 
        
    #TODO: update
    def get_predicate_space(self):
        all_tuples = []

        for v in Velocity:
            for r in Rotation:
                ##for cam_type in self.detection_cameras.append(0):
                    all_tuples.append((v,r))
        return all_tuples

