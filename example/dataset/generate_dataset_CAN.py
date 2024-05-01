import pandas as pd
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
import argparse
import numpy as np
from pathlib import Path
import utils


class NuScenesProcessor:
    def __init__(self, dataroot, version, dataoutput, key_frames=True, sensor="lidar", complexity=0):
        self.dataroot = dataroot
        self.version = version
        self.dataoutput = dataoutput
        self.key_frames = key_frames
        self.sensor = sensor
        self.complexity = complexity
        self.nuscenes = NuScenes(version, dataroot=Path(dataroot), verbose=True)
        self.nusc_can = NuScenesCanBus(dataroot=Path(dataroot))#'data/sets/nuscenes')
        #self.nusc_can.can_blacklist=[
        #    161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314
        #]

        if complexity == 1:
            self.cameras = ['CAM_FRONT', 'CAM_BACK']
        elif complexity == 2:
            self.cameras = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT']
        elif complexity == 3:
            self.cameras = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT','CAM_BACK_LEFT' ]
    
    
    '''
    def get_CAN_data(self):
        all_files = os.listdir(self.nusc_can.can_dir)
        scene_list = list(np.unique([f[:10] for f in all_files]))

        dataframes = []

        for scene_name in scene_list:
            steeranglefeedback_df = pd.DataFrame(self.nusc_can.get_messages(scene_name, 'steeranglefeedback', print_warnings=True))
            steeranglefeedback_df['name'] = scene_name

            dataframes.append(steeranglefeedback_df)

        combined_df = pd.concat(dataframes, ignore_index=True)

        return combined_df
    '''
    

    def cam_detection(self, samples: pd.DataFrame):
        """
        Given a sample in a scene, returns the objects in front of the vehicle, the action they are performing,
        and the visibility (0=min, 4=100%) from the ego-vehicle.

        NB:
        A sample of a scene (frame) has several sample annotations (Bounding Boxes). Each sample annotation
        has 0, 1, or + attributes (e.g., pedestrian moving, etc).
        The instance of an annotation is described in the instance table, which tracks the number of annotations
        in which the object appears.

        For each sample, check if there are any annotations. Retrieve the list of annotations for the sample.
        For each annotation, check from which camera it is from.

        """
        for sample_token in samples['sample_token']:
            sample = self.nuscenes.get('sample', sample_token)
            #detect_in_front = {cam_type: [] for cam_type in self.cameras}
            detected_objects = {cam_type: {} for cam_type in self.cameras}

            if len(sample['anns']) > 0:  # Check if sample has annotated objects
                for ann_token in sample['anns']:
                    for cam_type in self.cameras:
                        _, boxes, _ = self.nuscenes.get_sample_data(sample['data'][cam_type], box_vis_level=BoxVisibility.ANY,
                                                                    selected_anntokens=[ann_token])#, use_flat_vehicle_coordinates=True)
                        if len(boxes) > 0:
                            ann_info = self.nuscenes.get('sample_annotation', ann_token)
                            if len(ann_info['attribute_tokens']) > 0:
                                for attribute in ann_info['attribute_tokens']:
                                    attribute_name = self.nuscenes.get('attribute', attribute)['name']
                                    category = ann_info['category_name']
                                    #translation = np.array(self.nuscenes.get('translation', ann_info['translation']))
                                    #TODO: add obj_velocity = self.nuscenes.box_velocity(ann_token)
                                    visibility = int(self.nuscenes.get('visibility', ann_info['visibility_token'])['token'])
                                    if visibility >= 2:
                                        key = (category, attribute_name) #,visibility)
                                        if key not in detected_objects[cam_type]:
                                            detected_objects[cam_type][key] = 0
                                        detected_objects[cam_type][key] += 1 


            for cam_type in self.cameras:
                samples.loc[samples['sample_token'] == sample_token, f'detect_{cam_type}'] = str(detected_objects[cam_type])

        return samples


    def process_CAN_data(self):
        scene = pd.DataFrame(self.nuscenes.scene)[['token', 'name']].rename(columns={'token': 'scene_token'}) #add scene name, useful to merge the CAN BUS data 
        sample = pd.DataFrame(self.nuscenes.sample)[['token', 'timestamp', 'scene_token']].rename(columns={'token': 'sample_token', 'timestamp': 'utime'}).merge(scene, on='scene_token')

        merged_CAN_data_list = []

        valid_scene_names = [name for name in scene['name'] if int(name[-4:]) not in self.nusc_can.can_blacklist]

        for scene_name in valid_scene_names:
            steeranglefeedback_df = pd.DataFrame(self.nusc_can.get_messages(scene_name, 'steeranglefeedback', print_warnings=True))
            sample_scene = sample[sample['name'] == scene_name]
            sample_CAN_data = pd.merge_asof(sample_scene, steeranglefeedback_df, on='utime', direction='nearest', tolerance=10000)
            merged_CAN_data_list.append(sample_CAN_data)
            #TODO: add scenes with no CAN bus data

        merged_CAN_df = pd.concat(merged_CAN_data_list, ignore_index=True).drop(columns=['utime', 'name']).rename(columns={'value': 'steering_angle'})
        output_csv_path = Path(self.dataoutput) / 'can_data.csv'
        merged_CAN_df.to_csv(output_csv_path, index=False)
        print(f"Processed CAN data saved to {output_csv_path}")

    
    def process_scene_data(self) -> pd.DataFrame:
        """
        Processes agent data from the nuScenes dataset, creating a DataFrame with additional columns for velocity,
        acceleration, and heading change rate, based on sensor data filtering.

        Args:
            dataset (NuScenes): Instance of the NuScenes dataset.
            key_frame (bool): Flag indicating whether to filter for key frames only.
            sensor (str): Specific sensor modality to filter for.

        Returns:
            pd.DataFrame: A DataFrame containing processed scene data with dynamics calculations.
        """

        sample = pd.read_csv(Path(self.dataoutput) / 'can_data.csv')

        # add front camera objects information
        if self.complexity > 0:
            sample = self.cam_detection(sample)
        #complexity == 0 --> dataset without any knowledge of surrounding objects.


        sample_data = pd.DataFrame(self.nuscenes.sample_data).query(f"is_key_frame == {self.key_frames}")[['sample_token', 'ego_pose_token','calibrated_sensor_token']]
        #select only data related to selected sensor type (i.e. lidar)
        calibrated_sensors = pd.DataFrame(self.nuscenes.calibrated_sensor).rename(columns={'token': 'calibrated_sensor_token'})
        sensors = pd.DataFrame(self.nuscenes.sensor).rename(columns={'token': 'sensor_token'})
        sensors = sensors[sensors['modality'] == self.sensor].merge(calibrated_sensors, on='sensor_token').drop(columns=['rotation','translation', 'channel','camera_intrinsic', 'sensor_token'])
        merged_df = sensors.merge(sample_data, on='calibrated_sensor_token' ).drop(columns=['calibrated_sensor_token'])
               
        ego_pose = pd.DataFrame(self.nuscenes.ego_pose).rename(columns={'token': 'ego_pose_token'})
        ego_pose[['x', 'y', 'z']] = pd.DataFrame(ego_pose['translation'].tolist(), index=ego_pose.index)
    

        merged_df = sample.merge(merged_df, on='sample_token').merge(ego_pose, on='ego_pose_token').drop(columns=['ego_pose_token', 'sample_token', 'translation'])
        merged_df['yaw'] = merged_df['rotation'].apply(utils.quaternion_yaw).drop(columns=['rotation'])

        # Group by 'scene_token' and calculate dynamics
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], unit='us')
        merged_df.sort_values(by=['scene_token', 'timestamp'], inplace=True) #TODO: do I need it or is it already ordered?
        final_df = merged_df.groupby('scene_token', as_index=False).apply(utils.calculate_dynamics).dropna()

        # Compute  for each scene, the movement of the agent in local x and y
        final_df = pd.concat([utils.convert_coordinates(group) for _, group in final_df.groupby('scene_token')])

        # mark destination state
        #final_df['is_destination'] = False
        #for scene_token in final_df['scene_token'].unique():
        #    last_index = final_df[final_df['scene_token'] == scene_token].index[-1]
        #    final_df.at[last_index, 'is_destination'] = True

        return final_df


    def run_processing(self):
        self.process_CAN_data() #store file with processed CAN data
        states = self.process_scene_data()
        output_csv_path = Path(self.dataoutput) / f'dataset_{self.version}_{self.sensor}_{self.complexity}.csv'
        states.to_csv(output_csv_path, index=False)
        print(f"Processed data saved to {output_csv_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process nuScenes dataset and output processed data to CSV.")
    
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the nuScenes dataset directory.')
    parser.add_argument('--dataoutput', required=True, type=str, help='Path for the output CSV file directory.')
    parser.add_argument('--version', required=True, type=str, choices=["v1.0-mini", "v1.0-trainval"], help='Version of the nuScenes dataset to process.')  
    parser.add_argument('--key_frames', required=False, type=lambda x: (str(x).lower() == 'true'), default=True, help='Flag to process key frames only (True/False).')
    parser.add_argument('--sensor', required=False, type=str, default="lidar", choices=["lidar", "radar"], help='Specific sensor modality to filter for. Options: "lidar", "radar".')
    parser.add_argument('--complexity', required=True, type=int, default=0, choices=[0,1,2,3], help='Level of complexity of the dataset.')


    args = parser.parse_args()

    #DATAROOT = Path(args.dataroot) #'/data/sets/nuscenes'

    processor = NuScenesProcessor( args.dataroot, args.version, args.dataoutput, args.key_frames, args.sensor, args.complexity)
    processor.run_processing()



# Make the directory to store the nuScenes dataset in.
#!mkdir -p /data/sets/nuscenes

# Download the nuScenes mini split.
#!wget https://www.nuscenes.org/data/v1.0-mini.tgz

# Uncompress the nuScenes mini split.
#!tar -xf v1.0-mini.tgz -C ./data/sets/nuscenes

# Install nuScenes.
#!pip install nuscenes-devkit>=1.1.2 &> /dev/null #or conda install -n <--->

#create map/predictions
#paste prediciton_...
    
#load minidataset
#python3 generate_dataset_multisensor.py --dataroot 'data/sets/nuscenes' --version 'v1.0-mini' --dataoutput '.'


#load full dataset
#python3 generate_dataset_from_ego.py --dataroot /media/saramontese/Riccardo\ 500GB/NuScenesDataset/data/sets/nuscenes --version 'v1.0-trainval' --dataoutput /media/saramontese/Riccardo\ 500GB/NuScenesDataset/data/sets/nuscenes