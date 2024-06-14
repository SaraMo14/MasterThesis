import argparse
from pathlib import Path
import pandas as pd
import utils
from table_loader import BaseTableLoader
import time


class SceneDataProcessor(BaseTableLoader):
    """
        Processes agent data from the nuScenes dataset, creating a DataFrame with additional columns for velocity,
        acceleration, and heading change rate, based on sensor data filtering. 
        Merges data from CAN bus and Camera as well.

    """
    def __init__(self, dataroot, dataoutput, version, key_frames, sensor, complexity):
        super().__init__(dataroot, version)

        self.dataroot = dataroot
        self.dataoutput = dataoutput
        self.version = version
        self.key_frames = key_frames
        self.sensor = sensor
        self.complexity = complexity
        #self.nuscenes = NuScenes(version, dataroot=Path(dataroot), verbose=True)

        start_time = time.time()
        print("======\nLoading NuScenes tables for version {}...".format(self.version))

        self.sample_data = self.__load_table__('sample_data', drop_fields=['next', 'prev', 'width', 'height', 'filename','timestamp','fileformat'])
        print('sample_data loaded')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        print('calibrated_sensor loaded')
        self.sensors = self.__load_table__('sensor')
        print('sensor loaded')        
        self.ego_pose = self.__load_table__('ego_pose')     
        print('ego_pose loaded')
        #self.__make_reverse_index__(verbose=True)
        print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))




    def process_scene_data(self):
        sample = pd.read_csv(Path(self.dataoutput) / 'can_data.csv')
        #os.remove(Path(self.dataoutput) / 'can_data.csv')
        if self.complexity > 0:
            cam_data_path = Path(self.dataoutput) / 'cam_detection.csv'
            sample = pd.merge(sample, pd.read_csv(cam_data_path), on='sample_token')

        sample_data = pd.DataFrame(self.sample_data).query(f"is_key_frame == {self.key_frames}")[['sample_token', 'ego_pose_token', 'calibrated_sensor_token']]
        
        calibrated_sensors = pd.DataFrame(self.calibrated_sensor).rename(columns={'token': 'calibrated_sensor_token'})
        
        sensors = pd.DataFrame(self.sensors).rename(columns={'token': 'sensor_token'})
        sensors = sensors[sensors['modality'] == self.sensor].merge(calibrated_sensors, on='sensor_token').drop(columns=['rotation', 'translation', 'channel', 'camera_intrinsic', 'sensor_token'])

        merged_df = sensors.merge(sample_data, on='calibrated_sensor_token').drop(columns=['calibrated_sensor_token'])

        ego_pose = pd.DataFrame(self.ego_pose).rename(columns={'token': 'ego_pose_token'})
        ego_pose[['x', 'y', 'z']] = pd.DataFrame(ego_pose['translation'].tolist(), index=ego_pose.index)
        merged_df = sample.merge(merged_df, on='sample_token').merge(ego_pose, on='ego_pose_token').drop(columns=['ego_pose_token', 'translation'])
        
        merged_df['yaw'] = merged_df['rotation'].apply(utils.quaternion_yaw).drop(columns=['rotation'])

        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], unit='us')
        merged_df.sort_values(by=['scene_token', 'timestamp'], inplace=True)
        
        final_df = merged_df.groupby('scene_token', as_index=False).apply(utils.calculate_dynamics)#.dropna()

        final_df = pd.concat([utils.convert_coordinates(group) for _, group in final_df.groupby('scene_token')])
        return final_df

    def run_processing(self, test_size, random_state=42):
        states = self.process_scene_data()
        train_df, test_df = utils.train_test_split_by_scene(states, test_size, random_state)
        
        train_df.to_csv(Path(self.dataoutput) / f'train_{self.version}_{self.sensor}_{self.complexity}.csv', index=False)
        test_df.to_csv(Path(self.dataoutput) / f'test_{self.version}_{self.sensor}_{self.complexity}.csv', index=False)

        print(f"Dataset successfully saved to {self.dataoutput}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CAN and camera data, process scene data, and save the final dataset.")
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the nuScenes dataset directory.')
    parser.add_argument('--dataoutput', required=True, type=str, help='Path for the output data file directory.')
    parser.add_argument('--version', required=True, type=str, choices=["v1.0-mini", "v1.0-trainval"], help='Version of the nuScenes dataset to process.')
    parser.add_argument('--key_frames', required=True, type=bool, help='Whether to use key frames only.')
    parser.add_argument('--sensor', required=True, type=str, help='Sensor modality to process.')
    parser.add_argument('--complexity', required=True, type=int, default=0, choices=[0, 1, 2, 3], help='Level of complexity of the dataset.')
    parser.add_argument('--test_size', required=True, type=float, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', required=False, type=int, default=42, help='Random state for train-test split.')

    args = parser.parse_args()
    processor = SceneDataProcessor(args.dataroot, args.dataoutput, args.version, args.key_frames, args.sensor, args.complexity)
    processor.run_processing(args.test_size, args.random_state)

#python3 merge_and_process.py --dataroot 'data/sets/nuscenes' --dataoutput 'data/sets/nuscenes/' --version v1.0-trainval --key_frame True --sensor lidar --complexity 0 --test_size 0.2
