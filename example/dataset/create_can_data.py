import pandas as pd
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import argparse

from pathlib import Path


import os.path as osp
from table_loader import BaseTableLoader
import time


class CANDataProcessor(BaseTableLoader):
    
    def __init__(self, dataroot, dataoutput, version):
        super().__init__(dataroot, version)

        
        self.dataroot = dataroot
        self.dataoutput = dataoutput
        self.version = version
        self.nusc_can = NuScenesCanBus(dataroot=Path(dataroot))

        self.blacklist = [499, 515, 517, 501, 502, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314] #scenes with wrong route/ego poses
        
        

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        print("======\nLoading NuScenes tables for version {}...".format(self.version))

        self.log = self.__load_table__('log')
        print('log loaded')
        self.scene = self.__load_table__('scene')
        print('scene loaded')
        self.sample = self.__load_table__('sample')
        print('sample loaded')
        print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))


    def process_CAN_data(self):

        log_data = pd.DataFrame(self.log)[['token', 'location']].rename(columns={'token': 'log_token'})
        scene = pd.DataFrame(self.scene)[['token', 'name', 'log_token']].rename(columns={'token': 'scene_token'})
        sample = pd.DataFrame(self.sample)[['token', 'timestamp', 'scene_token']].rename(columns={'token': 'sample_token', 'timestamp': 'utime'}).merge(scene, on='scene_token')

        valid_scene_names = [name for name in scene['name'] if int(name[-4:]) not in self.blacklist]
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
    
    '''
    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table
    '''




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process nuScenes CAN data and save to a file.")
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the nuScenes dataset directory.')
    parser.add_argument('--dataoutput', required=True, type=str, help='Path for the output data file directory.')
    parser.add_argument('--version', required=True, type=str, choices=["v1.0-mini", "v1.0-trainval"], help='Version of the nuScenes dataset to process.')

    args = parser.parse_args()
    processor = CANDataProcessor(args.dataroot, args.dataoutput, args.version)
    processor.process_CAN_data()

    #/home/saramontese/Desktop/MasterThesis/example/dataset/create_can_data.py --dataroot /home/saramontese/Desktop/MasterThesis/example/dataset/data/sets/nuscenes --dataoutput /home/saramontese/Desktop/MasterThesis/example/dataset/data/sets/nuscenes --version v1.0-trainval