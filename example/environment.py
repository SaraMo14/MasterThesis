from typing import Any
from pgeon.environment import Environment
from example.discretizer.utils import BlockProgress, LanePosition
import pandas as pd    
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.map_expansion import arcline_path_utils
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
from example.dataset.utils import create_rotated_rectangle, calculate_direction_of_travel
from typing import Tuple, List

class SelfDrivingEnvironment(Environment):

    def __init__(self, city = "boston-seaport"):
        
        #self.stop_points = set() #list of destinations of scenes

        self.city = city
        self.nusc_map = NuScenesMap(dataroot='example/dataset/data/sets/nuscenes', map_name = city)
        self.dividers = getattr(self.nusc_map, 'road_divider') + getattr(self.nusc_map, 'lane_divider')

        self.current_state = None

        #self.threshold_distance = 8 #meters of distance within detecting traffic lights
        #TODO: change based on velocity, start slowing down before

    
    def reset(self, test_scenes: pd.DataFrame, seed = None) -> Any:
        
        #np.random.seed(seed)

        # Randomly select a scene_id
        unique_scene_ids = test_scenes['scene_token'].unique()
        selected_scene_id = np.random.choice(unique_scene_ids)
        
        # Get the starting point of the selected scene
        starting_point = test_scenes[test_scenes['scene_token'] == selected_scene_id][['x','y','velocity', 'yaw_rate', 'acceleration', 'yaw', 'steering_angle']].iloc[0].tolist()

        return starting_point#x, y, speed, yaw_rate, accel, yaw, steering_angle

    
        
    def step():
        pass

    

    #######################
    ### RENDERING
    #######################

    def render_egoposes_on_fancy_map(self, map_poses:list = [], 
                                     verbose: bool = True,
                                     #out_path: str = None,
                                     render_egoposes: bool = True,
                                     render_egoposes_range: bool = True,
                                     render_legend: bool = True):
        """
        Renders each ego pose of a trajectory on the map.
        
        :param map_poses: List of poses on the map.
        :param verbose: Whether to show status messages and progress bar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param render_egoposes: Whether to render ego poses.
        :param render_egoposes_range: Whether to render a rectangle around all ego poses.
        :param render_legend: Whether to render the legend of map layers.
        """

        explorer = NuScenesMapExplorer(self.nusc_map) #TODO: initialize in init()?

        # Settings
        patch_margin = 2
        min_diff_patch = 30

        # Ids of scenes with a bad match between localization and map.
        #scene_blacklist = [499, 515, 517]

        if verbose:
            print('Creating plot...')
        map_poses = np.vstack(map_poses)[:, :2]

        # Render the map patch with the current ego poses.
        min_patch = np.floor(map_poses.min(axis=0) - patch_margin)
        max_patch = np.ceil(map_poses.max(axis=0) + patch_margin)
        diff_patch = max_patch - min_patch
        if any(diff_patch < min_diff_patch):
            center_patch = (min_patch + max_patch) / 2
            diff_patch = np.maximum(diff_patch, min_diff_patch)
            min_patch = center_patch - diff_patch / 2
            max_patch = center_patch + diff_patch / 2
        my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])
        fig, ax = explorer.render_map_patch(my_patch, explorer.map_api.non_geometric_layers, figsize=(10, 10),
                                        render_egoposes_range=render_egoposes_range,
                                        render_legend=render_legend)

        # Plot in the same axis as the map.
        # Make sure these are plotted "on top".
        if render_egoposes:
            ax.scatter(map_poses[:, 0], map_poses[:, 1], s=20, c='k', alpha=1.0, zorder=2)
        plt.axis('off')

        #if out_path is not None:
        #plt.savefig(f'example/renderings/ego_poses_{datetime.now()}.png', bbox_inches='tight', pad_inches=0)
        #plt.close(fig)

        #return map_poses, fig, ax

    def render_ego_on_patch(self,x,y,yaw, patch_size=20, non_geometric_layers:List[str]=['road_divider', 'lane_divider']):

        patch_box = [x,y, patch_size, patch_size]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box)
        minx, miny, maxx, maxy = patch.bounds
        
        fig, ax = self.nusc_map.render_map_patch( [minx, miny, maxx, maxy], non_geometric_layers, figsize=(5, 5))
            
        ax.scatter(x,y)
        yaw =  math.degrees(-(math.pi / 2) + yaw)
        rotated_rectangle = create_rotated_rectangle((x,y), yaw, (2,4))
        x,y = rotated_rectangle.exterior.xy
        ax.plot(x,y,linewidth=0.2)
        

        # Plot the heading vector
        #ax.quiver(x, y, heading_vector[0], heading_vector[1], color='r', scale=5, label='Ego direction')

        # Plot the tangent vector
        #closest_point = lane_record[closest_pose_idx_to_lane]
        #ax.quiver(closest_point[0], closest_point[1], tangent_vector[0], tangent_vector[1], color='b', scale=5, label='Lane direction')
        #ax.legend()

        plt.title('Plot for ego vehicle direction')
        plt.show()   
    


    def render_rectangle_agent_scene(self, x,y,yaw, agent_size=(2,4)):
            #render agent as a rectangle and show its heading direction vector compared to the direction of the lane, for each sample in the scene.
            road_segment_token = self.nusc_map.record_on_point(x,y, 'road_segment')
            current_lane = self.nusc_map.record_on_point(x,y, 'lane')

            if road_segment_token and self.nusc_map.get('road_segment', road_segment_token)['is_intersection'] and not current_lane:
                closest_lane = self.nusc_map.get_closest_lane(x, y, radius=2)
                lane_path = self.nusc_map.get_arcline_path(closest_lane)
                closest_pose_idx_to_lane, lane_record, _ = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane_path)
                if closest_pose_idx_to_lane == len(lane_record) - 1:
                    tangent_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1]
                else:
                    tangent_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

            else:

                lane = self.nusc_map.get_arcline_path(current_lane)
                closest_pose_idx_to_lane, lane_record, distance_along_lane = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane)
                if closest_pose_idx_to_lane == len(lane_record) - 1:
                        tangent_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1]
                else:
                    tangent_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

    
              
            patch_size = 20
            patch_box = [x,y, patch_size, patch_size]
            patch = NuScenesMapExplorer.get_patch_coord(patch_box)
            minx, miny, maxx, maxy = patch.bounds

            fig, ax = self.nusc_map.render_map_patch( [minx, miny, maxx, maxy], ['road_divider', 'lane_divider'], figsize=(5, 5))
            
            ax.scatter(x,y)
            heading_vector = np.array([np.cos(yaw), np.sin(yaw)])

            yaw =  math.degrees(-(math.pi / 2) + yaw)
            rotated_rectangle = create_rotated_rectangle((x,y), yaw, agent_size)
            ax.quiver(x, y, heading_vector[0], heading_vector[1], color='b', scale=10, label='Ego Direction')
            ax.quiver(x,y, tangent_vector[0], tangent_vector[1],  color='r', scale=10, label='Lane Direction')
            x,y = rotated_rectangle.exterior.xy
            ax.plot(x,y)

    ########################
    ## PROCESS ENVIRONEMNT 
    ########################

    def process_environment(self, x, y, speed, yaw_rate, accel, yaw, steering_angle):
        """
        Processes environmental information to adjust vehicle behavior.

        Args:
            environment (dict): Information about drivable area, lanes, pedestrian crossings, etc.
        """

        #check if car is on drivable area, otherwise bring the car to the closest point on closest lane
        x,y = self.keep_drivable_area(x,y)

        #check if car is near a stop line 
        if self.is_near_stop_line(x, y):
            speed, yaw_rate, accel = 0, 0, 0
        
        #self.is_near_traffic_light(x,y)

        return x, y, speed, yaw_rate, accel, yaw, steering_angle


    def keep_drivable_area(self,x,y):
        is_lane_area = self.nusc_map.record_on_point(x, y, 'lane') 
        is_road = self.nusc_map.record_on_point(x, y, 'road_segment')
        #TODO: exclude road_blocks from road_segment
        if len(is_lane_area) == 0 and len(is_road)==0:
            x,y = self.reach_drivable_area(x,y)
        
        return x,y



    def is_near_traffic_light(self, x,y):
        """
        Check if there is a traffic light nearby the given pose (x, y).

        Args:
            x (float): Current x-coordinate of the vehicle.
            y (float): Current y-coordinate of the vehicle.

        Returns:
            dict: Information about the nearby traffic light (if any), or None.
        """
        for traffic_light in self.nusc_map.traffic_light:
            traffic_light_x = traffic_light['pose']['tx']
            traffic_light_y = traffic_light['pose']['ty']
            distance = np.sqrt((x - traffic_light_x)**2 + (y - traffic_light_y)**2)
            
            if distance <=self.threshold_distance:
                return self.get_traffic_light_status(traffic_light)
        
        return None
    


    def reach_drivable_area(self, x,y, radius:float=5, resolution_meters:float = 0.5):
        """
        Get closest lane id within a radius of query point. The distance from a point (x, y) to a lane is
        the minimum l2 distance from (x, y) to a point on the lane.
        Then, find the closest pose on this lane.
        Note that this function does not take the heading of the query pose into account.
        
        Args:
            x: X coordinate in global coordinate frame.
            y: Y Coordinate in global coordinate frame.
            radius: Radius around point to consider.        
            resolution_meters:How finely to discretize the lane.
        Return: 
            Tuple of the closest pose along the lane
        """

        lanes = self.nusc_map.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']
        discrete_points = self.nusc_map.discretize_lanes(lanes, resolution_meters) #returns ID, points for each lane

        current_min = np.inf
        closest_pose = (None, None)
        
        for lane_id, points in discrete_points.items():
            points_array = np.array(points)
            distances = np.linalg.norm(points_array[:, :2] - [x, y], axis=1)
            min_distance = distances.min()
            if min_distance <=current_min:
                current_min = min_distance
                closest_pose_index = distances.argmin()
                closest_pose = (points_array[closest_pose_index, 0], points_array[closest_pose_index, 1])

        return closest_pose
    

    def apply_brakes(self):
        pass

    def detect_pedestrian_crossing(self, x,y):
        pass
   
    def check_collision(self, state, object_type):
        pass

    #########################
    #PROCESS LANE INFORMATION
    #########################

    def is_near_stop_line(self,x,y):
        '''
        Function to check if car is near a stop line.
        Check also if a traffic light is nearby.
        #TODO: slow down gradually when approaching this parts of the environemnt.
        '''
        is_stop_line = self.nusc_map.record_on_point(x, y, 'stop_line')
        if len(is_stop_line)>0:

            return True #TODO: the action keeps the same, so if it was GAS it will keep being gas but with 0,0,0 state values
        
        return False
    
       
    def is_on_divider(self, x,y, yaw, agent_size:Tuple[float, float]) -> bool:
        """
        Checks whether the ego vehicle interescts lane and road dividers
        :param x,y,yaw: coordinates (in meters) and heading (in radians) of the agent 
        :param agent_size: height and width of the box representing the agent
        :return: True if agent intersects the layers specified in layer_name
        """
        yaw =  math.degrees(-(math.pi / 2) + yaw)
        
        # rectangle centered at (x, y) and rotated of yaw angle representing the agent
        rotated_rectangle = create_rotated_rectangle((x,y), yaw, agent_size)

        for record in self.dividers:
            line = self.nusc_map.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = rotated_rectangle.intersection(line)
            if not new_line.is_empty:
                return True

        return False  

    @staticmethod
    def project_pose_to_lane(pose, lane: List[arcline_path_utils.ArcLinePath], resolution_meters: float = 1):
        """
        Find the closest pose on a lane to a query pose and additionally return the
        distance along the lane for this pose. Note that this function does
        not take the heading of the query pose into account.
        :param pose: Query pose in (x,y) coordinates.
        :param lane: Will find the closest pose on this lane.
        :param resolution_meters: How finely to discretize the lane.
        :return: Tuple of the closest pose index and discretized xy points of the line.
        """

        discretized_lane = arcline_path_utils.discretize_lane(lane, resolution_meters=resolution_meters)

        xy_points = np.array(discretized_lane)[:, :2]
        closest_pose_index = np.linalg.norm(xy_points - pose[:2], axis=1).argmin()
        distance_along_lane = closest_pose_index *resolution_meters
        return closest_pose_index, xy_points, distance_along_lane
    



    def get_lane_info(self,x,y, yaw, eps=0.3, agent_size:Tuple[float, float]=(2,4)):
        """
        Determines the lane progress (which chunk the distance along the lane falls into). The lane is divided into 3 equal chunks.
        Determines the lane position (Left if on the left lane of the road, RIGHT is right and Center if in the center.)
        
             
        :return: (BlockProgress, LanePosition)
        """
        #NOTE: Lanes are not always straight. You should account for eventual curvature. 
        #print('Road objects on selected point:', nusc_map.layers_on_point(x, y), '\n')

        drivable_area = self.nusc_map.record_on_point(x,y, 'drivable_area')
        if not drivable_area:
            return (BlockProgress.NONE, LanePosition.NONE)
        
        road_segment_token = self.nusc_map.record_on_point(x,y, 'road_segment')
        current_lane = self.nusc_map.record_on_point(x,y, 'lane')

        if road_segment_token and self.nusc_map.get('road_segment', road_segment_token)['is_intersection'] and not current_lane:
            
            if self.is_on_divider(x,y, yaw, agent_size):
                return (BlockProgress.INTERSECTION, LanePosition.CENTER)
            else:
                closest_lane = self.nusc_map.get_closest_lane(x, y, radius=2)
                lane_path = self.nusc_map.get_arcline_path(closest_lane)
                closest_pose_idx_to_lane, lane_record, _ = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane_path)
                if closest_pose_idx_to_lane == len(lane_record) - 1:
                    tangent_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1]
                else:
                    tangent_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

                direction_of_travel = calculate_direction_of_travel(tangent_vector, yaw)

                if direction_of_travel <-eps:
                    return(BlockProgress.INTERSECTION, LanePosition.LEFT)
                elif direction_of_travel>eps:
                    return(BlockProgress.INTERSECTION,LanePosition.RIGHT)
                else:
                    return(BlockProgress.INTERSECTION,LanePosition.NONE) 
                    #TODO: fix tangent vector close to zero, meaning 2 points in the asrcline path are very close
                    #Possible fix: use prev or next points.
        
        block_progress = None
        lane_position = None
                
        #closest_lane = self.nusc_map.get_closest_lane(x, y, radius=2)                    
        lane = self.nusc_map.get_arcline_path(current_lane)
        closest_pose_idx_to_lane, lane_record, distance_along_lane = SelfDrivingEnvironment.project_pose_to_lane((x, y, yaw), lane)

        #1. Determine Block Progress
        chunk_size = arcline_path_utils.length_of_lane(lane) / 3

        if distance_along_lane < chunk_size:
             block_progress = BlockProgress.START 
        elif distance_along_lane < 2*chunk_size:
            block_progress = BlockProgress.MIDDLE
        else: 
            block_progress =  BlockProgress.END 



        #2. Determine Lane Position
        if self.is_on_divider(x,y, yaw, agent_size):
            lane_position = LanePosition.CENTER
        else:
            if closest_pose_idx_to_lane == len(lane_record) - 1:
                tangent_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1]
            else:
                tangent_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

            direction_of_travel = calculate_direction_of_travel(tangent_vector, yaw)
                    
            if direction_of_travel <-eps:
                #print("Opposite to travel direction of lane")
                lane_position = LanePosition.LEFT
            elif direction_of_travel>eps:
                #print("In travel direction of lane")
                lane_position = LanePosition.RIGHT
            else: 
                #print("Uncertain direction")
                lane_position = LanePosition.NONE #TODO: fix

        return (block_progress, lane_position)
        
            
    
    

    '''
    def get_direction_of_travel(self, x,y,yaw,  epsilon=0.3):
        #TODO: hanle intersections and other possible situations
        #1. compute closest point of arcline_path of current lane
        current_lane = self.nusc_map.record_on_point(x,y, 'lane')
        lane= self.nusc_map.get_arcline_path(current_lane)

        closest_pose_idx_to_lane, lane_record = self.project_pose_to_lane((x, y, yaw), lane)
        
        #2. compute reference direction vector
        #create a unit vector pointing in the direction of the lane's travel.ù
        #You can achieve this by taking two consecutive points from the arcline path and subtracting their positions.
        # Determine the tangent vector at the closest point
        if closest_pose_idx_to_lane == len(lane_record)-1 :
            tangent_vector = lane_record[closest_pose_idx_to_lane] - lane_record[closest_pose_idx_to_lane - 1] 
        else:
            tangent_vector = lane_record[closest_pose_idx_to_lane + 1] - lane_record[closest_pose_idx_to_lane]

        
        #3. NOrmalize vector to get a unit vector
        # Check if the tangent vector is a zero vector
        tangent_vector_norm = np.linalg.norm(tangent_vector)
        if tangent_vector_norm == 0:
            print("Uncertain direction due to zero tangent vector")
            return 0
        
        reference_direction_unit = tangent_vector / tangent_vector_norm  # Normalize the tangent vector

        #4.compute ego vehicle's heading direction vector
        heading_vector = np.array([np.cos(yaw), np.sin(yaw)])


        #5. compute dot product of there vectors. It ranges from -1 (completely opposite directions) to 1 (identical directions).
        dot_product = np.dot(reference_direction_unit, heading_vector)
         #6. Interpret results
        #A positive dot product (closer to 1) indicates the ego vehicle is aligned with the lane's travel direction.
        #A negative dot product (closer to -1) indicates the ego vehicle is facing the opposite direction of the lane.
        #A dot product near zero (-eps, +eps) suggests the vehicle is almost perpendicular to the lane's direction,

        if dot_product > epsilon:
            print("In travel direction of lane")
            return 1
        elif dot_product < -epsilon:
            print("Opposite to travel direction of lane")
            return -1
        else:
            print("Uncertain direction")
            return 0 #car perpendicular to the lane
        '''

    '''
    def get_lane_position(self, x,y, yaw, agent_size:Tuple[float, float]):
            """
            Returns Left, Right or Center based on the car position. 
            Left if on the left lane of the road, RIGHT is right and Center if in the 
            center.

            """

            #we suppose drivers stay on the right. 
            #1. check if has a road divider on the left. If yes: right

            #2. check if has a road divider on the right. If yes: left
            #3. check if ego vehicle is on road divider. If yes: center
            #4. right

            if self.is_on_divider(x,y, yaw, agent_size):
                    return LanePosition.CENTER
            #elif intersection
            else:
                direction = self.get_direction_of_travel(x,y,yaw)
                if direction>0:
                    return LanePosition.RIGHT
                elif direction<0:
                    return LanePosition.LEFT
                else:
                    return None #TODO: handle exceptions

    '''
    
    