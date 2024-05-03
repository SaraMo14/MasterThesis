from example.discretizer.utils import Position, Velocity, Rotation
from example.discretizer.discretizer import AVDiscretizer
from typing import Tuple, Union
import ast
from pgeon.discretizer import Predicate

# this class is a discretizer for discretizer Dxb DetectFront(0), DetectFront(person,walking)
#Discretizers D1b, D2b, D3b

class Detection:
    def __init__(self, category=None, attribute=None):
        self.category = category if category is not None else 0
        self.attribute = attribute if attribute is not None else 0
        
    def __str__(self) -> str:
        return f'{self.category},{self.attribute}'

    def __eq__(self, other):
        return self.category == other.category and self.attribute == other.attribute
    
    def __hash__(self):
        return hash((self.category, self.attribute))

class DetectInFront(Detection):
    pass

class DetectBack(Detection):
    pass

class DetectFrontRight(Detection):
    pass

class DetectFrontLeft(Detection):
    pass

class DetectRight(Detection):
    pass

class DetectLeft(Detection):
    pass





class AVDiscretizerD1b(AVDiscretizer):
    def __init__(self):
        super().__init__()
        self.DETECTION_CLASS_MAPPING = {
        'CAM_FRONT_LEFT': DetectFrontLeft,
        'CAM_FRONT_RIGHT': DetectFrontRight,
        'CAM_LEFT': DetectLeft,
        'CAM_RIGHT': DetectRight,
        'CAM_FRONT': DetectInFront,
        'CAM_BACK': DetectBack
        }
        self.STR_TO_CLASS_MAPPING = {
            'DetectFrontLeft': DetectFrontLeft,
            'DetectFrontRight': DetectFrontRight,
            'DetectLeft': DetectLeft,
            'DetectRight': DetectRight,
            'DetectInFront': DetectInFront,
            'DetectBack': DetectBack
        }
    
    def discretize_detections(self, detections):
        detected_predicates = []
        for cam_type, objects in detections.items():
            for (category, attribute), _ in ast.literal_eval(objects).items():

                #TODO: filter useful categories only.

                detection_class = self.DETECTION_CLASS_MAPPING.get(cam_type, None)
                if detection_class:
                    predicate = Predicate(
                        detection_class,
                        [detection_class(category, attribute)]
                    )
                    detected_predicates.append(predicate)
        return detected_predicates


    def str_to_state(self, state_str: str) -> Tuple[Union[Predicate, ]]:
        split_str = state_str.split('&')
        pos_str, vel_str, rot_str = split_str[0:3]
        x, y = map(int, pos_str[len("Position("):-1].split(','))
        
        mov_predicate = Velocity[vel_str[:-1].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-1].split('(')[1]]

        predicates = [
            Predicate(Position, [x, y]),
            Predicate(Velocity, [mov_predicate]),
            Predicate(Rotation, [rot_predicate])
        ]
        
        if len(split_str) > 3:
            detected_predicates = []
            for cam_detections in split_str[3:]:
                detection_class_str, cam_info = cam_detections[:-1].split('(')
                category, attribute = map(str, cam_info.split(','))
                detection_class = self.STR_TO_CLASS_MAPPING.get(detection_class_str, None)
                detected_predicates.append(Predicate(detection_class, [detection_class(category, attribute)]))
            predicates.extend(detected_predicates)

        return tuple(predicates)




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



    #TODO: update
    def get_predicate_space(self):
        all_tuples = []

        for v in Velocity:
            for r in Rotation:
                ##for cam_type in self.detection_cameras.append(0):
                    all_tuples.append((v,r))
        return all_tuples

