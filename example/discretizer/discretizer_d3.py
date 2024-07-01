from example.discretizer.utils import Velocity, Rotation, IsTrafficLightNearby,IsZebraNearby,IsStopSignNearby, IsPedestrianNearby, IsBikeNearby
from example.discretizer.discretizer_d2 import AVDiscretizerD2
from example.environment import SelfDrivingEnvironment
from pgeon.discretizer import Predicate
from example.discretizer.utils import LanePosition, BlockProgress, NextIntersection, Velocity, Rotation
import ast


import numpy as np
from typing import Tuple, Union

class AVDiscretizerD3(AVDiscretizerD2):
    def __init__(self, environment: SelfDrivingEnvironment):
        super().__init__(environment)
        self.environment = environment
    ##################################
    ### New Predicates and Discretizers
    ##################################

    def discretize_vulnerable_subjects(self, state_detections):
        #since there are not many cases we discretize into YES/NO rather than 0/1-2/2+
        n_peds, n_bikes = self.is_vulnerable_subject_nearby(state_detections)
        is_ped_nearby = IsPedestrianNearby.YES  if n_peds > 0 else IsPedestrianNearby.NO
        is_bike_nearby = IsBikeNearby.YES  if n_bikes > 0 else IsBikeNearby.NO

        return is_ped_nearby, is_bike_nearby


    @staticmethod
    def is_vulnerable_subject_nearby(state_detections):

        """
        Function to check for vulnerable subjects nearby based on state detections from all cameras.
        Vulnerable subjects include pedestrians, cyclists and people driving scooters.
        
        Parameters:
            state_detections (dict): A dictionary where keys are camera types and values are 
                                    serialized object detection results in string format at a given state.
        
        Returns:
            tuple: A tuple containing the total count of pedestrians and cyclists.
        
        NOTE: Only subjects with >40% visibility are considered.
        """
        tot_ped_count = 0
        tot_bike_count = 0
        for cam_type, objects in state_detections.items():

                for (category, attribute), count in ast.literal_eval(objects).items():
                    if 'human.pedestrian' in category:
                        if 'personal_mobility' in category and 'with_rider' in attribute: #scooter
                            tot_bike_count+=count
                        else:
                            tot_ped_count +=count
                    elif 'bicycle' in category and 'with_rider' in attribute:
                        tot_bike_count +=count

                return tot_ped_count, tot_bike_count


    ##################################
    ### Overridden Methods
    ##################################

    
    def discretize(self, state: np.ndarray, detections=None) -> Tuple[Predicate, ...]:
        predicates = super().discretize(state, detections)

        pedestrian_predicate, bike_predicate = self.discretize_vulnerable_subjects(detections)
        return (Predicate(IsPedestrianNearby, [pedestrian_predicate]), Predicate(IsBikeNearby,[bike_predicate]), ) + predicates

    
    def str_to_state(self, state_str: str) -> Tuple[Union[Predicate, ]]:
        split_str = state_str.split(' ')
        
        pedestrian_str, bike_str, stop_sign_str, zebra_str, traffic_light_str, block_str, lane_pos_str, next_inter_str, vel_str, rot_str = split_str[0:10]

        pedestrian_predicate = IsPedestrianNearby[pedestrian_str[:-2].split('(')[2]] 
        bike_predicate = IsBikeNearby[bike_str[:-2].split('(')[1]] 
        stop_sign_predicate = IsStopSignNearby[stop_sign_str[:-2].split('(')[1]] 
        zebra_predicate = IsZebraNearby[zebra_str[:-2].split('(')[1]] 
        traffic_light_predicate = IsTrafficLightNearby[traffic_light_str[:-2].split('(')[1]] 
        progress_predicate = BlockProgress[block_str[:-2].split('(')[1]] 
        position_predicate = LanePosition[lane_pos_str[:-2].split('(')[1]]
        intersection_predicate = NextIntersection[next_inter_str[:-2].split('(')[1]]
        mov_predicate = Velocity[vel_str[:-2].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-2].split('(')[1]]

        predicates = [
            Predicate(IsPedestrianNearby, [pedestrian_predicate]),
            Predicate(IsBikeNearby, [bike_predicate]),
            Predicate(IsStopSignNearby, [stop_sign_predicate]),
            Predicate(IsZebraNearby, [zebra_predicate]),
            Predicate(IsTrafficLightNearby, [traffic_light_predicate]),
            Predicate(BlockProgress, [progress_predicate]),
            Predicate(LanePosition, [position_predicate]),
            Predicate(NextIntersection, [intersection_predicate]),
            Predicate(Velocity, [mov_predicate]),
            Predicate(Rotation, [rot_predicate])
        ]
        

        if len(split_str) > 10:
            detected_predicates = []
            for cam_detections in split_str[8:]:
                
                detection_class_str, count = cam_detections[:-1].split('(')
                detection_class = self.STR_TO_CLASS_MAPPING.get(detection_class_str, None)
                detected_predicates.append(Predicate(detection_class, [detection_class(count)]))
            
            predicates.extend(detected_predicates)

        return tuple(predicates)



   

    #TODO: update
    def nearest_state(self, state):
        pass


    
    def get_predicate_space(self):
        #predicate_space = super().get_predicate_space()
        
        # Append new predicates to the existing space
        #new_predicate_space = [(p,l,n,v,r,d) for p,l,n,v,r in self.get_predicate_space() for d in self.NEW_PREDICATE_VALUES]
        # return predicate_space + new_predicate_space
        pass