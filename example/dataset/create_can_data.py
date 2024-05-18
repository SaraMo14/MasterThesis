import pandas as pd
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes import NuScenes
import argparse

from pathlib import Path

class CANDataProcessor:
    
    def __init__(self, dataroot, dataoutput, version):
        self.dataroot = dataroot
        self.dataoutput = dataoutput
        self.version = version
        self.nuscenes = NuScenes(version, dataroot=Path(dataroot), verbose=True)
        self.nusc_can = NuScenesCanBus(dataroot=Path(dataroot))

    def process_CAN_data(self):
        log_data = pd.DataFrame(self.nuscenes.log)[['token', 'location']].rename(columns={'token': 'log_token'})
        scene = pd.DataFrame(self.nuscenes.scene)[['token', 'name', 'log_token']].rename(columns={'token': 'scene_token'})
        sample = pd.DataFrame(self.nuscenes.sample)[['token', 'timestamp', 'scene_token']].rename(columns={'token': 'sample_token', 'timestamp': 'utime'}).merge(scene, on='scene_token')

        valid_scene_names = [name for name in scene['name'] if int(name[-4:]) not in self.nusc_can.can_blacklist]
        merged_CAN_data_list = []

        for scene_name in valid_scene_names:
            steeranglefeedback_df = pd.DataFrame(self.nusc_can.get_messages(scene_name, 'steeranglefeedback', print_warnings=True))
            sample_scene = sample[sample['name'] == scene_name]
            sample_CAN_data = pd.merge_asof(sample_scene, steeranglefeedback_df, on='utime', direction='nearest', tolerance=10000)
            merged_CAN_data_list.append(sample_CAN_data)

        merged_CAN_df = pd.concat(merged_CAN_data_list, ignore_index=True).drop(columns=['utime', 'name']).rename(columns={'value': 'steering_angle'}).merge(log_data, on='log_token').drop(columns=['log_token'])
        output_path = Path(self.dataoutput) / 'can_data.csv'
        merged_CAN_df.to_csv(output_path, index=False)
        print(f"Processed CAN data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process nuScenes CAN data and save to a file.")
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the nuScenes dataset directory.')
    parser.add_argument('--dataoutput', required=True, type=str, help='Path for the output data file directory.')
    parser.add_argument('--version', required=True, type=str, choices=["v1.0-mini", "v1.0-trainval"], help='Version of the nuScenes dataset to process.')

    args = parser.parse_args()
    processor = CANDataProcessor(args.dataroot, args.dataoutput, args.version)
    processor.process_CAN_data()