from example.discretizer.utils import Velocity, Rotation, IsTrafficLightNearby,IsZebraNearby,IsStopSignNearby
from example.discretizer.discretizer import AVDiscretizer
#from example.discretizer.discretizer_d1 import AVDiscretizerD1
from example.environment import SelfDrivingEnvironment
from pgeon.discretizer import Predicate
from example.discretizer.utils import LanePosition, BlockProgress, NextIntersection, Velocity, Rotation

import numpy as np
from example.dataset.utils import create_rectangle

from typing import Tuple, Union

class AVDiscretizerD2(AVDiscretizer):#AVDiscretizerD1):
    def __init__(self, environment: SelfDrivingEnvironment):
        super().__init__(environment)
        self.environment = environment
    ##################################
    ### New Predicates and Discretizers
    ##################################


    def discretize_stop_line(self, x,y,yaw):
        # Create a rotated rectangle around the vehicle's current pose
        yaw_in_deg = np.degrees(-(np.pi / 2) + yaw)
        area = create_rectangle((x,y), yaw_in_deg, size=(15,20), shift_distance=10)

        is_sign_nearby = IsStopSignNearby.YES  if self.environment.is_near_stop_sign(x,y,area) else IsStopSignNearby.NO
        
        is_zebra_nearby = IsZebraNearby.YES  if self.environment.is_near_ped_crossing(area) else IsZebraNearby.NO
        
        is_traffic_light_nearby = IsTrafficLightNearby.YES  if self.environment.is_near_traffic_light(yaw, area) else IsTrafficLightNearby.NO

        return is_sign_nearby, is_zebra_nearby, is_traffic_light_nearby


    ##################################
    ### Overridden Methods
    ##################################

    
    def discretize(self, state: np.ndarray, detections=None) -> Tuple[Predicate, ...]:
        predicates = super().discretize(state, detections)
        x, y, _, _, yaw = state 
        stop_sign_predicate, zebra_predicate, traffic_light_predicate = self.discretize_stop_line(x,y,yaw)

        return (Predicate(IsStopSignNearby, [stop_sign_predicate]), Predicate(IsZebraNearby,[zebra_predicate]), Predicate(IsTrafficLightNearby, [traffic_light_predicate])) + predicates

    
    def str_to_state(self, state_str: str) -> Tuple[Union[Predicate, ]]:
        split_str = state_str.split(' ')
        
        stop_sign_str, zebra_str, traffic_light_str, block_str, lane_pos_str, next_inter_str, vel_str, rot_str = split_str[0:8]

        stop_sign_predicate = IsStopSignNearby[stop_sign_str[:-2].split('(')[2]] 
        zebra_predicate = IsZebraNearby[zebra_str[:-2].split('(')[1]] 
        traffic_light_predicate = IsTrafficLightNearby[traffic_light_str[:-2].split('(')[1]] 
        progress_predicate = BlockProgress[block_str[:-2].split('(')[1]] 
        position_predicate = LanePosition[lane_pos_str[:-2].split('(')[1]]
        intersection_predicate = NextIntersection[next_inter_str[:-2].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-2].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-2].split('(')[1]]

        predicates = [
            Predicate(IsStopSignNearby, [stop_sign_predicate]),
            Predicate(IsZebraNearby, [zebra_predicate]),
            Predicate(IsTrafficLightNearby, [traffic_light_predicate]),
            Predicate(BlockProgress, [progress_predicate]),
            Predicate(LanePosition, [position_predicate]),
            Predicate(NextIntersection, [intersection_predicate]),
            Predicate(Velocity, [mov_predicate]),
            Predicate(Rotation, [rot_predicate])
        ]
        

        if len(split_str) > 8:
            detected_predicates = []
            for cam_detections in split_str[8:]:
                
                detection_class_str, count = cam_detections[:-1].split('(')
                detection_class = self.STR_TO_CLASS_MAPPING.get(detection_class_str, None)
                detected_predicates.append(Predicate(detection_class, [detection_class(count)]))
            
            predicates.extend(detected_predicates)

        return tuple(predicates)


    #TODO: 2 states are near in terms of detected objects if:
        #1. they have the same amount of detected objects in each camera (done)
        #2. if there are any cameras where both states have detected objects > 0.
         #they both have detected objects in at least one common camera, regarless of the amount OR 
            #they both not have detected objects in front or in back
    def nearest_state(self, state):
        pass


    #TODO: update
    def get_predicate_space(self):
        #predicate_space = super().get_predicate_space()
        
        # Append new predicates to the existing space
        #new_predicate_space = [(p,l,n,v,r,d) for p,l,n,v,r in self.get_predicate_space() for d in self.NEW_PREDICATE_VALUES]
        # return predicate_space + new_predicate_space
        pass