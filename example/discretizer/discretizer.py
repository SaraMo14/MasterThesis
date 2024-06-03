from example.dataset.utils import vector_angle
from example.discretizer.utils import Action, LanePosition, BlockProgress, NextIntersection, Velocity, Rotation, DetectedObject
from example.environment import SelfDrivingEnvironment
import numpy as np
from typing import Tuple, List, Union
from pgeon.discretizer import Discretizer, Predicate

# this class is a discretizer for discretizers D0 (no cameras), D1A (generic and with 2 cameras)

class AVDiscretizer(Discretizer):
    def __init__(self, environment: SelfDrivingEnvironment):
        super(AVDiscretizer, self).__init__()

        self.enviroment = environment

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
        
        self.agent_size = (2,4)#(1.730, 4.084) #width, length in metres

    
    
    
    @staticmethod
    def is_close(a, b, eps=0.1):
        return abs(a - b) <= eps

    ##################################
    ### DISCRETIZERS
    ##################################

    def discretize(self,
                   state: np.ndarray, detections=None
                   ) -> Tuple[Predicate, ...]:
        x, y, velocity, steer_angle, yaw = state 
        block_progress_pred, lane_pos_pred  = self.discretize_position(x,y,yaw)
        mov_predicate = self.discretize_speed(velocity)
        rot_predicate = self.discretize_steering_angle(steer_angle)

        if detections is not None:
            detected_predicates = self.discretize_detections(detections)
            return (Predicate(BlockProgress, [block_progress_pred]),
                    Predicate(LanePosition, [lane_pos_pred]),
                    Predicate(NextIntersection, [NextIntersection.NONE]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]),
                *detected_predicates)
        else:
            return (Predicate(BlockProgress, [block_progress_pred]),
                    Predicate(LanePosition, [lane_pos_pred]),
                    Predicate(NextIntersection, [NextIntersection.NONE]),
                Predicate(Velocity, [mov_predicate]),
                Predicate(Rotation, [rot_predicate]))
        

    def discretize_detections(self, detections)-> List:
        detected_predicates = []
        for cam_type, objects in detections.items():
            obj = DetectedObject() if objects=="{}" else DetectedObject(cam_type) 
            
            detected_predicates.append(Predicate(DetectedObject, [obj]))
        return detected_predicates


    def discretize_steering_angle(self, steering_angle: float)->Rotation:
        for i, threshold in enumerate(self.rotation_thr):
            if steering_angle <= threshold:  
                return Rotation(i + 1)
        return Rotation.LEFT  


    def discretize_position(self, x,y,yaw)-> Tuple[BlockProgress,LanePosition]:

        block_progress, lane_position = self.enviroment.get_lane_info(x,y, yaw, eps=0.3, agent_size=self.agent_size)
        
        return block_progress, lane_position
    


    def discretize_speed(self, speed) -> Velocity:
        for i, threshold in enumerate(self.velocity_thr):
            if speed <= threshold: 
                return Velocity(i + 1)
        return Velocity.VERY_HIGH
    
    
        
    def assign_intersection_actions(trajectory, intersection_info):
        """
        Assigns actions based on intersection information.

        Args:
            trajectory: List containing the discretized trajectory.
            intersection_info: List storing information about intersections.

        Returns:
            Updated trajectory with assigned actions for intersections.
        """
        for i in range(0, len(trajectory), 2):  # Access states
            if trajectory[i][0] == Predicate(BlockProgress, BlockProgress.INTERSECTION):
                print(f'frame {i/2} --> {trajectory[i]}')
                continue

            for idx, action in intersection_info:
                if 2 * idx > i and 2 * idx < len(trajectory) - 1:
                    state = list(trajectory[i])
                    state[2] = action
                    trajectory[i] = tuple(state)
                    break
            print(f'frame {i/2} --> {trajectory[i]}')
        return trajectory
    

    @staticmethod
    def determine_intersection_action(start_position, end_position) -> NextIntersection:
        """
        Determine the action at the intersection based on positional changes.
        
        Args:
            start_position (x,y,x1,y1): vector containing starting position just before the intersection and at the beginning of the intersection.
            end_position (x,y,x1,y1): vector containin position at the intersection and just after the intersection.
        Returns:
            Action: NextIntersection.RIGHT, NextIntersection.LEFT, or NextIntersection.STRAIGHT.
        """


        x_1,y_1, x_2, y_2 = start_position
        x_n,y_n, x_n1, y_n1 = end_position

        #Calculate the movement vector from point (x1, y1) to point (x2, y2)
        pre_vector = np.array([x_2 - x_1, y_2 - y_1]) 
        post_vector = np.array([x_n1 - x_n, y_n1 - y_n]) 
        angle = vector_angle(pre_vector, post_vector)

        #Determine action based on the angle
        if abs(angle) < np.radians(30):
            return Predicate(NextIntersection,[NextIntersection.STRAIGHT])
        elif np.cross(pre_vector, post_vector) > 0:
            return Predicate(NextIntersection,[NextIntersection.LEFT])
        else:
            return Predicate(NextIntersection,[NextIntersection.RIGHT])



    ##################################
    




    def determine_action(self, current_state, next_state) -> Action:
        delta_x0, delta_y0, vel_t,  acc_t0, steer_t0 = current_state
        delta_x1, delta_y1, vel_t1,  acc_t1, steer_t1 = next_state

        if self.is_close(delta_x1, 0, self.eps_pos_x) and self.is_close(delta_y1, 0, self.eps_pos_y) and self.is_close(vel_t, 0, self.eps_vel):
            return Action.IDLE
        #if delta_y1 < -self.eps_pos_y:
            #return Action.REVERSE

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
        split_str = state_str.split(' ')
        block_str, pos_str, vel_str, rot_str = split_str[0:4]

        progress_predicate = BlockProgress[block_str[:-2].split('(')[1]]
        position_predicate = LanePosition[pos_str[:-2].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-2].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-2].split('(')[1]]

        if split_str[4:]:
            detected_predicates = []
            for cam_detections in split_str[3:]:
                cam_type = cam_detections[:-1].split('(')[1]#map(str.strip, cam_detections[:-2].split(','))
                detected_predicates.append(Predicate(DetectedObject, [DetectedObject(cam_type)]))
            return (Predicate(BlockProgress, [progress_predicate]),
                        Predicate(LanePosition, [position_predicate]),
                        Predicate(Velocity, [mov_predicate]),
                        Predicate(Rotation, [rot_predicate]), *detected_predicates)
        else: 
            return (Predicate(BlockProgress, [progress_predicate]),
                    Predicate(LanePosition, [position_predicate]),
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

        for p in BlockProgress:
            for l in LanePosition:
                for n in NextIntersection:
                    for v in Velocity:
                        for r in Rotation:
                            ##for cam_type in self.detection_cameras.append(0):
                            all_tuples.append((p,l,n,v,r))
        return all_tuples

