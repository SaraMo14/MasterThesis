import argparse
from pathlib import Path
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

class CamDataProcessor:
    
    def __init__(self, dataroot, dataoutput, version, complexity):
        self.dataroot = dataroot
        self.dataoutput = dataoutput
        self.version = version
        self.complexity = complexity
        self.nuscenes = NuScenes(version, dataroot=Path(dataroot), verbose=True)

        if complexity == 1:
            self.cameras = ['CAM_FRONT', 'CAM_BACK']
        elif complexity == 2:
            self.cameras = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT']
        elif complexity == 3:
            self.cameras = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']

    def cam_detection(self, sample_tokens:pd.DataFrame):
        """
        Given a sample in a scene, returns the objects in front of the vehicle, the action they are performing,
        and the visibility (0=min, 4=100%) from the ego-vehicle.

        NOTE:
        A sample of a scene (frame) has several sample annotations (Bounding Boxes). Each sample annotation
        has 0, 1, or + attributes (e.g., pedestrian moving, etc).
        The instance of an annotation is described in the instance table, which tracks the number of annotations
        in which the object appears.

        For each sample, check if there are any annotations. Retrieve the list of annotations for the sample.
        For each annotation, check from which camera it is from.

        """
        #detected_objects = []
        for sample_token in sample_tokens['token']:
            sample = self.nuscenes.get('sample', sample_token)
            sample_detected_objects = {cam_type: {} for cam_type in self.cameras}

            if sample['anns']: #if sample has annotated objects
                for ann_token in sample['anns']:
                    for cam_type in self.cameras:
                        _, boxes, _ = self.nuscenes.get_sample_data(
                            sample['data'][cam_type], 
                            box_vis_level=BoxVisibility.ANY, 
                            selected_anntokens=[ann_token]
                        )
                        if boxes:
                            ann_info = self.nuscenes.get('sample_annotation', ann_token)
                            if ann_info['attribute_tokens']:
                                for attribute in ann_info['attribute_tokens']:
                                    attribute_name = self.nuscenes.get('attribute', attribute)['name']
                                    category = ann_info['category_name']
                                    #TODO: add obj_velocity = self.nuscenes.box_velocity(ann_token)
                                    
                                    visibility = int(self.nuscenes.get('visibility', ann_info['visibility_token'])['token'])
                                    if visibility >= 2:
                                        key = (category, attribute_name)
                                        if key not in sample_detected_objects[cam_type]:
                                            sample_detected_objects[cam_type][key] = 0
                                        sample_detected_objects[cam_type][key] += 1

            #detected_objects.append((sample_token, sample_detected_objects))
            for cam_type in self.cameras:
                sample_tokens.loc[sample_tokens['token'] == sample_token, f'{cam_type}'] = str(sample_detected_objects[cam_type])

        
        df_detected_objects = pd.DataFrame(sample_tokens).rename(columns={'token': 'sample_token'})#, columns=['sample_token', 'detected_objects'])
        output_path = Path(self.dataoutput) / 'cam_detection.csv'
        df_detected_objects.to_csv(output_path, index=False)
        print(f"Camera detection data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process nuScenes camera data and save to a file.")
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the nuScenes dataset directory.')
    parser.add_argument('--dataoutput', required=True, type=str, help='Path for the output data file directory.')
    parser.add_argument('--version', required=True, type=str, choices=["v1.0-mini", "v1.0-trainval"], help='Version of the nuScenes dataset to process.')
    parser.add_argument('--complexity', required=True, type=int, default=0, choices=[0, 1, 2, 3], help='Level of complexity of the dataset.')

    args = parser.parse_args()
    processor = CamDataProcessor(args.dataroot, args.dataoutput, args.version, args.complexity)
    sample_tokens = pd.DataFrame(processor.nuscenes.sample)['token'].to_frame()
    processor.cam_detection(sample_tokens)

