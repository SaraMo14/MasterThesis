from typing import Optional, Any
from pgeon.environment import Environment
from example.discretizer.utils import Velocity, Action, Rotation
import pandas as pd    
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.map_expansion import arcline_path_utils
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class SelfDrivingEnvironment(Environment):

    def __init__(self, city = "boston-seaport"):
        
        self.stop_points = set() #list of destinations of scenes

        self.city = city
        self.nusc_map = NuScenesMap(dataroot='example/dataset/data/sets/nuscenes', map_name = city)


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

    

    def compute_reward(self, current_state, action):
        """
        Computes the reward for transitioning from the current_state to next_state via action.

        Args:
            current_state: discretized current velocity, position and rotation action,
            action:
            next_state: discretized next velocity, position and rotation action

        Return:
            tuple: final reward components Tuple(float)
        
        Ref: https://arxiv.org/pdf/2405.01440
        Ref: CIRL: controllable imitative reinforcement learning for vision-based self-driving 
        """
        #TODO: rewards between -1,0,1
        #TODO: add objects

        speed_reward = 0
        safety_reward = 0
        smoothness_reward = 0
        progress_reward = 0

        
        vel_predicate = current_state[1].value[0]
        velocity = Velocity[str(vel_predicate)[:-1].split('(')[1]]
        steer_predicate = current_state[2].value[0]
        steer_angle = Rotation[str(steer_predicate)[:-1].split('(')[1]]

        # Speed-based reward and penalty
        if velocity in [Velocity.LOW, Velocity.MEDIUM, Velocity.HIGH]:
            speed_reward += 50
        else:  # Velocity.VERY_HIGH
            speed_reward -= 50  # Penalize very high speeds for safety reasons


        #TODO: modify to handle nearby objects
        # Penalize stopping in potentially unsafe or unnecessary situations (when car is in the back for example)
        #if velocity == Velocity.STOPPED:
        #    safety_reward -= 0.1
        #if object_back and velocity.stopped --> penalize
        
        # Collision penalties NOTE: should it be next state or current state?
        #if self.check_collision(current_state, 'vehicle') or self.check_collision(current_state, 'pedestrian'):
            #safety_reward -= 100
        #elif self.check_collision(current_state, 'object'):
            #safety_reward -= 50

        # • penalty for steering angles in ranges assumed incorrect for current command (e.g., going left during a turn-right command)
        if action in [Action.GAS_TURN_LEFT, Action.TURN_LEFT] and steer_angle in [Rotation.RIGHT, Rotation.SLIGHT_RIGHT]:
            smoothness_reward-=20
        if action is [Action.GAS_TURN_RIGHT, Action.TURN_RIGHT] and steer_angle in [Rotation.LEFT, Rotation.SLIGHT_LEFT]:
            smoothness_reward-=20

       
        # Overlapping with opposite-direction lane penalty or upon leaving the lane (if distance from center of lane is greater than 2)
        #if self.check_overlap(current_state, 'opposite_lane'):
            #safety_reward -= 100
        
        #if self.check_overlap(current_state, 'sidewalk'):
            #safety_reward -= 100


        if action in [Action.GAS, Action.BRAKE]:
            smoothness_reward -= 10
        """
        if action in [Action.TURN_LEFT, Action.TURN_RIGHT, Action.GAS_TURN_LEFT, Action.GAS_TURN_RIGHT, Action.BRAKE_TURN_LEFT, Action.BRAKE_TURN_RIGHT]:
            smoothness_reward -= 0.1
        elif action == Action.STRAIGHT:
            smoothness_reward += 0.1
        elif  action in [Action.GAS, Action.BRAKE]:
            smoothness_reward += 0.05
        """

        # Penalty for high-speed turns
        if velocity in [Velocity.HIGH, Velocity.VERY_HIGH]:
            if steer_angle in [Rotation.LEFT, Rotation.RIGHT]:
                smoothness_reward -= 50
            elif steer_angle in [Rotation.SLIGHT_LEFT, Rotation.SLIGHT_RIGHT]:
                smoothness_reward -= 20
            
        
        # to encourage actions that lead to progress towards a goal, give positive reward if current state is a intermediate destination
        if current_state not in self.stop_points:
            progress_reward += 10
        else:
            progress_reward -=1
        
        return speed_reward, safety_reward, smoothness_reward, progress_reward
        
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
        plt.savefig(f'example/renderings/ego_poses_{datetime.now()}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        #return map_poses, fig, ax

    
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

    def is_near_stop_line(self,x,y):
        '''
        Function to check if car is near a stop line.
        Check also if a traffic light is nearby.
        #TODO: slow down graually when approaching this parts of the environemnt.
        '''
        is_stop_line = self.nusc_map.record_on_point(x, y, 'stop_line')
        if len(is_stop_line)>0:

            return True #TODO: the action keeps the same, so if it was GAS it will keep being gas but with 0,0,0 state values
        
        return False
            

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
    

    def get_lane_position(self, x,y):
        """
        Returns Left, Right or Center based on the car position. Left if on the left lane of the road, RIGHT is right and Center if in the 
        center.

        """
        pass



    def get_lane_block_position(self,x,y, yaw):
        """
        Determine which chunk the distance along the lane falls into.
        The lane is divided into 3 equal chunks.
        
        :param distance_along_lane: The distance along the lane.
        :param total_lane_length: The total length of the lane.
        
        :return: The chunk number (1, 2, or 3). 0 for intersection and lane connections and -1 for other cases
        """
        #NOTE: Lanes are not always straight. You should account for eventual curvature. 
        #print('Road objects on selected point:', nusc_map.layers_on_point(x, y), '\n')
        drivable_area = self.nusc_map.record_on_point(x,y, 'drivable_area')
        if drivable_area:
            road_segment_token = self.nusc_map.record_on_point(x,y, 'road_segment')
            current_lane = self.nusc_map.record_on_point(x,y, 'lane')

            if road_segment_token and self.nusc_map.get('road_segment', road_segment_token)['is_intersection'] and not current_lane:
                return 0
            else:
                #closest_lane = self.nusc_map.get_closest_lane(x, y, radius=2)                    
                lane_record = self.nusc_map.get_arcline_path(current_lane)
                _, distance_along_lane = arcline_path_utils.project_pose_to_lane((x, y, yaw), lane_record)
                lane_length = arcline_path_utils.length_of_lane(lane_record)

                        #print(f'current lane: {current_lane}')
                        #print(f'distance_along_lane {distance_along_lane}')
                        #print(f'lane length: {lane_length}')
                        #print()

                chunk_size = lane_length / 3

                if distance_along_lane < chunk_size:
                    return 1
                elif distance_along_lane < 2*chunk_size:
                    return 2
                else: 
                    return 3

        else:
            return -1 #On car parking, walkway, pedestrian crossings


    def apply_brakes(self):
        # Implement braking logic
        pass

    def detect_pedestrian_crossing(self, x,y):
        # Implement logic to detect pedestrian crossings
        pass

    def steer_to_center_lane(self, lanes):
        # Implement logic to steer towards the center of the current lane
        pass

    def avoid_lane_dividers(self, lane_dividers):
        # Implement logic to avoid lane dividers
        pass

   
    # Helper functions to check for collisions and overlaps
    def check_collision(self, state, object_type):
        # Implement collision checking logic here
        pass



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
