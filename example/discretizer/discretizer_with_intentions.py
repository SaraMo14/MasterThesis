from example.discretizer.utils import Position, Velocity, Rotation
from example.discretizer.discretizer import AVDiscretizer
from typing import Tuple, Union
import ast
from pgeon.discretizer import Predicate

# this class is a discretizer for discretizer D1a

#Discretizers D1, D2, D3
class Detection:
    chunks = ["0", "1-4", "5+"]

    def __init__(self, count=0):
        if isinstance(count, str):
            self.count = count
        elif count == 0:
            self.count = self.chunks[0]
        elif count < 5:
            self.count = self.chunks[1]
        else:
            self.count = self.chunks[2]
    
    def __str__(self) -> str:
        return f'{self.count}'

    def __eq__(self, other):
        return self.count == other.count
    
    def __hash__(self):
        return hash(self.count)

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



class AVDiscretizerD1(AVDiscretizer):
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
            tot_count = 0
            for (category, _), count in ast.literal_eval(objects).items():

                #filter useful categories only (no movable or static objects)
                if 'object' not in category:
                    tot_count+=count
            detection_class = self.DETECTION_CLASS_MAPPING.get(cam_type, None)
            predicate = Predicate(
                    detection_class,
                    [detection_class(tot_count)]
            )
            detected_predicates.append(predicate)
        return detected_predicates


    def str_to_state(self, state_str: str) -> Tuple[Union[Predicate, ]]:
        split_str = state_str.split(' ')
        pos_str, vel_str, rot_str = split_str[0:3]
        x, y = map(int, pos_str[len("(Position("):-2].split(','))
        
        mov_predicate = Velocity[vel_str[:-2].split('(')[1]]
        rot_predicate = Rotation[rot_str[:-2].split('(')[1]]

        predicates = [
            Predicate(Position, [x, y]),
            Predicate(Velocity, [mov_predicate]),
            Predicate(Rotation, [rot_predicate])
        ]
        
        if len(split_str) > 3:
            detected_predicates = []
            for cam_detections in split_str[3:]:
                
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
        og_position, og_velocity, og_angle, *detections = state

        x,y = og_position.value

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
        
 
        if len(detections) ==2:
                for front_chunk in Detection.chunks:
                    for back_chunk in Detection.chunks:
                        front_obj, back_obj = DetectInFront(front_chunk), DetectBack(back_chunk)
                        if (front_obj, back_obj) not in detections:
                            yield og_position, og_velocity, og_angle, Predicate(DetectInFront, [front_obj]), Predicate(DetectBack, [back_obj])
        #TODO: implement
        elif len(detections) == 4:
            pass
         
        elif len(detections) == 6:
            pass


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
                    
                    #TODO: implement
                    if len(detections) ==2: 
                        for front_chunk in Detection.chunks:
                            for back_chunk in Detection.chunks:
                                front_obj, back_obj = DetectInFront(front_chunk), DetectBack(back_chunk)
                                if (front_obj, back_obj) not in detections:
                                    yield Predicate(Position, [new_x, new_y]), og_velocity, og_angle, Predicate(DetectInFront, [front_obj]), Predicate(DetectBack, [back_obj])
                    #TODO: implement
                    elif len(detections) == 4:
                        pass
                    
                    elif len(detections) == 6:
                        pass


    #TODO: update
    def get_predicate_space(self):
        all_tuples = []

        for v in Velocity:
            for r in Rotation:
                ##for cam_type in self.detection_cameras.append(0):
                    all_tuples.append((v,r))
        return all_tuples

