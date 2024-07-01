from example.dataset.utils import vector_angle
from example.discretizer.utils import Detection, FrontLeftObjects, FrontObjects, FrontRightObjects, BackLeftObjects, BackRightObjects, FrontObjects, BackObjects, Action, LanePosition, BlockProgress, NextIntersection, Velocity, Rotation
from example.environment import SelfDrivingEnvironment
import numpy as np
from typing import Tuple, Union
import ast
from pgeon.discretizer import Discretizer, Predicate

# this class is a discretizer for discretizers D0 (no cameras), D1 (with 2 cameras)

class AVDiscretizer(Discretizer):
    def __init__(self, environment: SelfDrivingEnvironment):
        #super(AVDiscretizer, self).__init__()
        super().__init__()

        self.enviroment = environment

        self.velocity_thr = [0.2, 7.3, 11.1,  14.3] #[0.2, 6, 11, 17]#m/s 
        #self.yaw_thr = [-2*np.pi/3, -np.pi/3, np.pi/3, 2*np.pi/3]  #[-2.5, -1, 0., 1, 2.5] #radiants
        self.rotation_thr = [-2.4, -0.3, 0.3, 3.2]# [-3, -0.3, 0.3 , 3]
        self.eps_rot = 0.4
        self.eps_vel = 0.2 #0.5 
        self.eps_acc = 0.3

        self.frequency = 0.5 #2Hz
        
        self.agent_size = (1.730, 4.084) #width, length in meters

        self.state_to_be_discretized = ['x', 'y', 'velocity', 'steering_angle', 'yaw'] 
        self.state_columns_for_action = ['velocity', 'acceleration', 'steering_angle']     

        self.DETECTION_CLASS_MAPPING = {
        'CAM_FRONT_LEFT': FrontLeftObjects,
        'CAM_FRONT_RIGHT': FrontRightObjects,
        'CAM_BACK_LEFT': BackLeftObjects,
        'CAM_BACK_RIGHT': BackRightObjects,
        'CAM_FRONT': FrontObjects,
        'CAM_BACK': BackObjects
        }
        self.STR_TO_CLASS_MAPPING = {
            'FrontLeftObjects': FrontLeftObjects,
            'FrontRightObjects': FrontRightObjects,
            'BackLeftObjects': BackLeftObjects,
            'BackRightObjects': BackRightObjects,
            'FrontObjects': FrontObjects,
            'BackObjects': BackObjects
        }
    
    @staticmethod
    def is_close(a, b, eps=0.1):
        return abs(a - b) < eps

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
        

    def discretize_detections(self, detections):
        detected_predicates = []
        for cam_type, objects in detections.items():
            tot_count = 0
            for (category, _), count in ast.literal_eval(objects).items():

                #TODO: filter useful categories only.
                if 'object' not in category:
                    tot_count+=count
            detection_class = self.DETECTION_CLASS_MAPPING.get(cam_type, None)
            predicate = Predicate(
                    detection_class,
                    [detection_class(tot_count)]
            )
            detected_predicates.append(predicate)
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
    
    
        
    def assign_intersection_actions(self,trajectory, intersection_info, verbose = False):
        """
        Assigns actions based on intersection information.

        Args:
            trajectory: List containing the discretized trajectory.
            intersection_info: List storing information about intersections as.

        Returns:
            Updated trajectory with assigned actions for intersections.
        """
        for i in range(0, len(trajectory), 2):  # Access states
            
                
            '''
            block_progress = next((predicate for predicate in trajectory[i] if predicate.predicate.__name__ == 'BlockProgress'), None)

            if block_progress == Predicate(BlockProgress, BlockProgress.INTERSECTION):
                if verbose:
                    print(f'frame {int(i/2)} --> {trajectory[i]}')
                    if i<len(trajectory) - 1:
                        print(f'action: {self.get_action_from_id(trajectory[i+1])}')
                    else:
                        print('END')
                continue

            '''

            for idx, action in intersection_info:
                if 2 * idx > i and 2 * idx < len(trajectory) - 1: #check if the intersection state (2*idx) comes next the current state (i)
                    state = list(trajectory[i])
                    next_intersect_idx = next((i for i, predicate in enumerate(state) if predicate.predicate.__name__ == 'NextIntersection'), None)
                    state[next_intersect_idx] = action
                    trajectory[i] = tuple(state)
                    break
            if verbose:
                    print(f'frame {int(i/2)} --> {trajectory[i]}')
                    if i<len(trajectory) - 1:
                        print(f'action: {self.get_action_from_id(trajectory[i+1])}')
                    else:
                        print('END')

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
    




    def determine_action(self, next_state) -> Action:
        vel_t1,  acc_t1, steer_t1 = next_state
        if vel_t1 < self.eps_vel and self.is_close(acc_t1,0,self.eps_acc):
            return Action.IDLE
    
        # determine acceleration
        if acc_t1 >= self.eps_acc and vel_t1>=self.eps_vel:
            acc_action = Action.GAS
        elif acc_t1 <= -self.eps_acc and vel_t1>=self.eps_vel:
            acc_action = Action.BRAKE
        else:
            acc_action = None

        # determine direction
        if steer_t1 <= -self.eps_rot:#_r:#delta_x1 > self.eps_pos_x: #TODO : steer_t1 or t0?
            dir_action = Action.TURN_RIGHT
        elif steer_t1 >= self.eps_rot:#_l:#delta_x1 < -self.eps_pos_x:
            dir_action = Action.TURN_LEFT
        else:
            dir_action = Action.STRAIGHT

       # Combine acceleration and direction actions
        if acc_action == Action.GAS:
            if dir_action == Action.TURN_RIGHT:
                return Action.GAS_TURN_RIGHT
            elif dir_action == Action.TURN_LEFT:
                return Action.GAS_TURN_LEFT
            else:
                return Action.GAS
        elif acc_action == Action.BRAKE:
            if dir_action == Action.TURN_RIGHT:
                return Action.BRAKE_TURN_RIGHT
            elif dir_action == Action.TURN_LEFT:
                return Action.BRAKE_TURN_LEFT
            else:
                return Action.BRAKE
        elif acc_action is None:
            # Fallback to direction if no acceleration action was determined
            return dir_action

        #NOTE: we do not consider Action.REVERSE

        # if no other conditions met
        return Action.STRAIGHT



    def _discretize_state_and_action(self, scene, i):
        #Given a scene from the dataset, it discretizes the current state (i) and determines the  following action.
        current_state_to_discretize = scene.iloc[i][self.state_to_be_discretized].tolist()
        current_detection_info = scene.iloc[i][self.detection_cameras] if self.detection_cameras else None
        discretized_current_state = self.discretize(current_state_to_discretize, current_detection_info)

        next_state_for_action = scene.iloc[i+1][self.state_columns_for_action].tolist()
        action = self.determine_action(next_state_for_action)
        action_id = self.get_action_id(action)
        
        return discretized_current_state, action_id

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
        block_str, lane_pos_str, next_inter_str, vel_str, rot_str = split_str[0:5]

        progress_predicate = BlockProgress[block_str[:-2].split('(')[2]] 
        position_predicate = LanePosition[lane_pos_str[:-2].split('(')[1]]
        intersection_predicate = NextIntersection[next_inter_str[:-2].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-2].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-2].split('(')[1]]

        predicates = [
            Predicate(BlockProgress, [progress_predicate]),
            Predicate(LanePosition, [position_predicate]),
            Predicate(NextIntersection, [intersection_predicate]),
            Predicate(Velocity, [mov_predicate]),
            Predicate(Rotation, [rot_predicate])
        ]
        

        if len(split_str) > 5:
            detected_predicates = []
            for cam_detections in split_str[5:]:
                
                detection_class_str, count = cam_detections[:-1].split('(')
                detection_class = self.STR_TO_CLASS_MAPPING.get(detection_class_str, None)
                detected_predicates.append(Predicate(detection_class, [detection_class(count)]))
            
            predicates.extend(detected_predicates)

        return tuple(predicates)

    
    
    def nearest_state(self, state):
        og_block_progress, og_lane_position, og_next_intersection, og_velocity, og_angle, *detections = state

        #NOTE: the order of the following conditions affects the yielded Predicates, thus introducing bias.
        # Generate nearby positions considering discretization
        for b in BlockProgress:
            if b != og_block_progress.value:
                yield Predicate(BlockProgress, b),og_lane_position, og_next_intersection, og_velocity, og_angle, *detections
        
        for l in LanePosition:
            if l != og_lane_position.value:
                yield og_block_progress, Predicate(LanePosition, l), og_next_intersection, og_velocity, og_angle, *detections

        for n in NextIntersection:
            if n != og_next_intersection.value:
                yield og_block_progress, og_lane_position, Predicate(NextIntersection, n), og_velocity, og_angle, *detections

        for v in Velocity:
            if v != og_velocity.value:
                yield og_block_progress, og_lane_position, Predicate(Velocity, v), og_angle, *detections
        
        for r in Rotation:
            if r != og_angle.value:
                yield og_block_progress, og_lane_position, og_velocity, Predicate(Rotation, r), *detections
        if detections:
            similar_detections = []
            for cam_type in self.detection_cameras:      
                detection_class = self.STR_TO_CLASS_MAPPING.get(cam_type, None)
                for chunk in Detection.chunks:         
                    similar_detections.append(Predicate(detection_class, [chunk])) #TODO: this will yield also a state equal to the current. To be fixed, Further idea: 0 is similar to 1-4 but not to 5+.
            #print(*similar_detections)
            print(f'detections: ')
            print(*detections)
            print(f'detection 0 value;: {detections[0].value}')
            yield og_block_progress, og_lane_position, og_velocity, og_velocity, og_angle, *similar_detections         
        

        for b in BlockProgress:
            for l in LanePosition:
                for n in NextIntersection:
                    for v in Velocity:
                        for r in Rotation:
                            amount_of_equals_to_og = \
                                int(b == og_block_progress.value) + int(l == og_lane_position.value) + int(n == og_next_intersection.value) + int(v == og_velocity.value) + int(r==og_angle.value)
                            if amount_of_equals_to_og <=3:
                                yield Predicate(BlockProgress, b), Predicate(LanePosition, l), Predicate(NextIntersection, n), Predicate(Velocity, v), Predicate(Rotation, r), *detections

                        #TODO: fix
                        #if len(detections)>0:
                        #    for objs in [(DetectedObject('CAM_FRONT'), DetectedObject()),
                        #        (DetectedObject('CAM_FRONT'), DetectedObject('CAM_BACK')),
                        #        (DetectedObject(), DetectedObject('CAM_BACK'))]:
                        #        if objs not in detections:
                        #            yield Predicate(Position, [new_x, new_y]), og_velocity, og_angle, Predicate(DetectedObject, [objs[0]]), Predicate(DetectedObject, [objs[1]])



    def all_actions(self):
        return list(Action) 
        
    #TODO: update with detections
    def get_predicate_space(self):
        all_tuples = []

        for p in BlockProgress:
            for l in LanePosition:
                for n in NextIntersection:
                    for v in Velocity:
                        for r in Rotation:
                            #for cam_type in self.detection_cameras:#.append(0):
                                #all_tuples.append((p,l,n,v,r))
                                all_tuples.append((p,l,n,v,r))
        return all_tuples

