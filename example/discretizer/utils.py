from enum import Enum, auto

class Velocity(Enum):
  STOPPED = auto()
  LOW = auto()
  MEDIUM = auto()
  HIGH = auto()
  VERY_HIGH = auto()

  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class Rotation(Enum):
  RIGHT = auto()
  SLIGHT_RIGHT = auto()
  FORWARD = auto()
  SLIGHT_LEFT = auto()
  LEFT = auto()

  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class LanePosition(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()
    NONE = auto() #for all the cases not includend in the previous categories (e.g car headed perpendicular to the road, parkins, etc..)
    #TODO: handle intersections    
    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'

class BlockProgress(Enum):
    START = auto()
    MIDDLE = auto()
    END = auto()
    INTERSECTION = auto()
    NONE = auto() #for all the cases not includend in the previous categories (e.g. car parkings, walkway)

    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'


class NextIntersection(Enum):
    #what is my behavior at the next intersection? Do i go left, STRAIGH or RIGHT?
    # This is something that you know being a driver of the vehicle and affects how you're going to drive.
    # This allows the check for desires.
    RIGHT = auto()
    LEFT = auto()
    STRAIGHT = auto()
    IDLE = auto()
    NONE = auto()

    def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


#Discretizer D1a
class DetectedObject(): #0 or camera_type if any object is present
    def __init__(self, cam_type=None):
        self.cam_type = cam_type if cam_type is not None else 0

    def __str__(self) -> str:
        return f'{self.cam_type}'

    def __eq__(self, other):
        return self.cam_type == other.cam_type 
    
    def __hash__(self):
        return hash(self.cam_type)




class Action(Enum):
  IDLE = auto() 
  TURN_LEFT = auto()
  TURN_RIGHT = auto()
  GAS = auto() 
  BRAKE = auto()
  #REVERSE = auto()
  STRAIGHT = auto() #car keep going straight at same pace
  GAS_TURN_RIGHT= auto()
  GAS_TURN_LEFT= auto()
  BRAKE_TURN_RIGHT = auto()  
  BRAKE_TURN_LEFT = auto()
  #TODO:differentiate between sharp and slight accelaraion, slight turn, ..., lane keeping, preparing to lane change, and lane changing (more of intentations)



