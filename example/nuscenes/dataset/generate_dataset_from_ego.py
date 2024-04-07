import pandas as pd
from nuscenes import NuScenes
from nuscenes.prediction import convert_global_coords_to_local
import argparse
import numpy as np
from pathlib import Path
import utils


def convert_coordinates(group: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the global coordinates of an object in each row to local displacement  relative to its previous position and orientation.

    This function iterates through a DataFrame where each row represents an object's state at a given time, including its position (x, y) and orientation (rotation).
    It computes the local displacement (delta_local_x, delta_local_y) at each timestep, using the position and orientation from the previous timestep as the reference frame.

    Args:
        group (DataFrame): A DataFrame containing the columns 'x', 'y' (global position coordinates), and 'rotation' (object's orientation as a quaternion).

    Returns:
        DataFrame: The input DataFrame with two new columns added ('delta_local_x' and 'delta_local_y') that represent the local displacement relative to the
                   previous position and orientation.

    Note:
        The first row of the output DataFrame will have 'delta_local_x' and 'delta_local_y'
        set to 0.0, as there is no previous state to compare.
    """

    #traslation = (group.iloc[0]['x'], group.iloc[0]['y'], 0) # Using the first row as the origin
    #rotation = group.iloc[0]['rotation']
    #coordinates = group[['x','y']].values
    #local_coords = convert_global_coords_to_local(coordinates, traslation, rotation)

    #group['local_x'], group['local_y'] = local_coords[:, 0], local_coords[:, 1]

    # Initialize the displacement columns for the first row
    group['delta_local_x'], group['delta_local_y'] = 0.0, 0.0

    for i in range(1, len(group)):
        # Use the previous row's position as the origin for translation
        translation = (group.iloc[i-1]['x'], group.iloc[i-1]['y'], 0)
        
        # Use the previous row's rotation; assuming constant rotation for simplicity
        rotation = group.iloc[i-1]['rotation']
        
        # Current row's global coordinates
        coordinates = group.iloc[i][['x', 'y']].values.reshape(1, -1)
        
        # Convert global coordinates to local based on the previous row's state
        local_coords = convert_global_coords_to_local(coordinates, translation, rotation)
        
        # Update the DataFrame with the computed local displacements
        group.at[group.index[i], 'delta_local_x'], group.at[group.index[i], 'delta_local_y'] = local_coords[0, 0], local_coords[0, 1]
    
    return group


def calculate_dynamics(group: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates velocity, acceleration, and heading change rate for each entry in a DataFrame,
    assuming the DataFrame is sorted by timestamp. The function adds three new columns to the
    DataFrame: 'velocity', 'acceleration', and 'heading_change_rate'.

    Args:
        group (DataFrame): A pandas DataFrame containing at least 'timestamp', 'x', 'y', and 'yaw'
                           columns. 'timestamp' should be in datetime format, and the DataFrame should
                           be sorted based on 'timestamp'.
    
    Returns:
        DataFrame: The input DataFrame with three new columns added: 'velocity', 'acceleration', and
                   'heading_change_rate', representing the calculated dynamics.
                   
    Note:
        This function handles cases where consecutive timestamps might be identical (time_diffs == 0)
        by avoiding division by zero and setting the respective dynamics values to NaN.
    """
    time_diffs = group['timestamp'].diff().dt.total_seconds()
    
    # Handle potential division by zero for velocity and acceleration calculations
    valid_time_diffs = time_diffs.replace(0, np.nan)
    
    # Calculate displacement (Euclidean distance between consecutive points)
    displacements = group[['x', 'y']].diff().pow(2).sum(axis=1).pow(0.5)
    
    # Meters / second.
    group['velocity'] = displacements / valid_time_diffs
    
    # Meters / second^2.
    group['acceleration'] = group['velocity'].diff() / valid_time_diffs
    
    # Radians / second.
    group['heading_change_rate'] = group['yaw'].diff() / valid_time_diffs

    return group


def process_scene_data(dataset: NuScenes, key_frame: bool, sensor: str) -> pd.DataFrame:
    """
    Processes agent data from the nuScenes dataset, creating a DataFrame with additional columns for velocity,
    acceleration, and heading change rate, based on sensor data filtering.

    Args:
        dataset (NuScenes): Instance of the NuScenes dataset.
        key_frame (bool): Flag indicating whether to filter for key frames only.
        sensor (str): Specific sensor modality to filter for. Use 'all' to include all sensors.

    Returns:
        pd.DataFrame: A DataFrame containing processed scene data with dynamics calculations.
    """

    sample = pd.DataFrame(dataset.sample)[['token', 'scene_token']].rename(columns={'token': 'sample_token'})
    sample_data = pd.DataFrame(dataset.sample_data).query(f"is_key_frame == {key_frame}")[['sample_token', 'ego_pose_token','calibrated_sensor_token']]
    ego_pose = pd.DataFrame(dataset.ego_pose).rename(columns={'token': 'ego_pose_token'})
    ego_pose[['x', 'y', 'z']] = pd.DataFrame(ego_pose['translation'].tolist(), index=ego_pose.index)
    
    merged_df = sample.merge(sample_data, on='sample_token').merge(ego_pose, on='ego_pose_token').drop(columns=['ego_pose_token', 'sample_token', 'translation'])

    if sensor != 'all':
        calibrated_sensors = pd.DataFrame(dataset.calibrated_sensor).rename(columns={'token': 'calibrated_sensor_token'})
        sensors = pd.DataFrame(dataset.sensor).rename(columns={'token': 'sensor_token'})
        sensors = sensors[sensors['modality'] == sensor].merge(calibrated_sensors, on='sensor_token').drop(columns=['rotation','translation', 'channel','camera_intrinsic', 'sensor_token'])
        merged_df = sensors.merge(merged_df, on='calibrated_sensor_token' ).drop(columns=['calibrated_sensor_token'])
        
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], unit='us')

    merged_df['yaw'] = merged_df['rotation'].apply(utils.quaternion_yaw)

    merged_df.sort_values(by=['scene_token', 'timestamp'], inplace=True)

    # Group by 'scene_token' and calculate dynamics
    final_df = merged_df.groupby('scene_token', as_index=False).apply(calculate_dynamics).dropna()

    # Compute  for each scene, the movement of the agent in local x and y
    df_updated = pd.concat([convert_coordinates(group) for _, group in final_df.groupby('scene_token')])

    return df_updated


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process nuScenes dataset and output processed data to CSV.")
    
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the nuScenes dataset directory.')
    parser.add_argument('--dataoutput', required=True, type=str, help='Path for the output CSV file directory.')
    parser.add_argument('--version', required=True, type=str, help='Version of the nuScenes dataset to process.') #v1.0-mini, v1.0-trainval, etc. 
    parser.add_argument('--key_frames', required=False, type=lambda x: (str(x).lower() == 'true'), default=True, help='Flag to process key frames only (True/False).')
    parser.add_argument('--sensor', required=False, type=str, default="lidar", choices=["all", "lidar", "camera", "radar"], help='Specific sensor modality to filter for. Options: "all", "lidar", "camera", "radar".')



    args = parser.parse_args()

    DATAROOT = Path(args.dataroot) #'/data/sets/nuscenes'

    nuscenes = NuScenes(args.version, dataroot=DATAROOT, verbose=True)

    states = process_scene_data(nuscenes, args.key_frames, args.sensor)

    DATAOUTPUT = Path(args.dataoutput)
    output_csv_path = DATAOUTPUT / 'dataset_from_ego.csv'
    states.to_csv(output_csv_path, index=False) 
    print(f"Processed data saved to {output_csv_path}")





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
#python3 generate_dataset_from_ego.py --dataroot 'data/sets/nuscenes' --version 'v1.0-mini'


#load full dataset
#python3 generate_dataset_from_ego.py --dataroot /media/saramontese/Riccardo\ 500GB/NuScenesDataset/data/sets/nuscenes --version 'v1.0-trainval' --dataoutput /media/saramontese/Riccardo\ 500GB/NuScenesDataset/data/sets/nuscenes