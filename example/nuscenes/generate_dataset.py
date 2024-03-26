import pandas as pd
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
import argparse

from pathlib import Path

def process_agent_data(mini_train, helper):
  """
  Processes agent data from mini_train list and helper object, creating a DataFrame
  with additional columns for velocity, acceleration, heading change rate, and future behavior.

  Args:
      mini_train (list): List of strings representing agent IDs in "instance_token_sample_token" format.
      helper (PredictHelper): An instance of the PredictHelper class for data access.

  Returns:
      pandas.DataFrame: A DataFrame containing agent data with additional columns.
  """

  data = []  # Empty list to store all data efficiently

  for row in mini_train:
    instance_token, sample_token = row.split("_")

    # Get annotation and store for later
    annotation = helper.get_sample_annotation(instance_token, sample_token)
    print(annotation)
    yaw, pitch, roll = quaternion_to_euler(annotation['rotation'])
    #yaw_degrees = math.degrees(yaw)
    # Get future agent behavior
    #future_local_xy = helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=True)
    
    # Add current data to the temporary list
    data.append({
        'instance_token': instance_token,
        'sample_token': sample_token,
        #'token': annotation['token'],
        'translation': annotation['translation'],
        'yaw': yaw, #<0 if car is turning left, >0 if turning right
        #'pitch': pitch,
        #'roll': roll,
        'velocity': helper.get_velocity_for_agent(instance_token, sample_token),
        'acceleration': helper.get_acceleration_for_agent(instance_token, sample_token),
        'heading_change_rate': helper.get_heading_change_rate_for_agent(instance_token, sample_token),
        #'prev': annotation['prev'],
        #'next': annotation['next']
    })

  df = pd.DataFrame(data)
  return df


def fill_acceleration_na(df):
    """
    Fills missing values in the 'acceleration' column of a DataFrame based on velocity differences.

    Args:
        df (pandas.DataFrame): The DataFrame containing agent data.

    Returns:
        pandas.DataFrame: The DataFrame with missing values in 'acceleration' column filled.
    """
    # Calculate the acceleration based on velocity differences
    velocity_diff = df['velocity'].diff() / 0.5

    # Replace NaN values with 0 if there is no previous row or different instance_token
    df['acceleration'] = df.apply(lambda row: 0 if pd.isna(row['acceleration'])
                                  or pd.isna(velocity_diff.iloc[row.name - 1])
                                  or row['instance_token'] != df.at[row.name - 1, 'instance_token']
                                  else velocity_diff.iloc[row.name - 1], axis=1)

    # Fill remaining NaN values with 0
    df['acceleration'].fillna(0, inplace=True)

    return df


import math

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to Euler angles (yaw, pitch, roll).
    
    Args:
        quaternion (list): List representing the quaternion [w, x, y, z].
        
    Returns:
        tuple: Euler angles (yaw, pitch, roll) in radians.
    """
    w, x, y, z = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use ±π/2 if |sinp| >= 1
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp) # using  yaw_rad = np.arctan2(2 * (x * y + w * z), 1 - 2 * (x**2 + y**2)) is more efficient?
    

    return yaw, pitch, roll #in radiants 



'''
def fillna_ffill_bfill(df, column):
    """
    Fills missing values in a DataFrame column using forward fill (ffill) and backward fill (bfill).

    Args:
        df (pandas.DataFrame): The DataFrame containing the column with missing values.
        column (str): The name of the column to fill.

    Returns:
        pandas.DataFrame: A new DataFrame with missing values filled using ffill or bfill.
    """

    # Forward fill (ffill)
    df[column] = df[column].fillna(method='ffill')

    # Backward fill (bfill) if any NaN values remain
    if df[column].isnull().any():
        df[column].fillna(method='bfill', inplace=True)

    return df
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, type=str)
    parser.add_argument('--version', required=True, type=str)
    parser.add_argument('--split', required=True, type=str)
    args = parser.parse_args()


    # This is the path where you stored your copy of the nuScenes dataset.
    DATAROOT = Path(args.dataroot) #'/data/sets/nuscenes'

    # Initialize nuScenes instance.
    nuscenes = NuScenes(args.version, dataroot=DATAROOT)#'v1.0-mini', dataroot=DATAROOT)

    # Get mini train split for prediction challenge.
    mini_train = get_prediction_challenge_split(args.split, dataroot=DATAROOT)

    # Initialize PredictHelper.
    helper = PredictHelper(nuscenes)

    # Process agent data and create DataFrame.
    df = process_agent_data(mini_train, helper)

    # Handle missing values in the DataFrame.
    df = fill_acceleration_na(df)

    # Merge with sample data.
    df_sample = pd.DataFrame(nuscenes.sample).rename(columns={'token': 'sample_token'}).drop(columns=["prev", "next", "data", "anns"]).rename(columns={'token': 'sample_token'})
    states_df = pd.merge(df, df_sample, left_on='sample_token', right_on='sample_token', how='left')#.drop(columns=["sample_token"])

    # Output the processed DataFrame to a CSV file.
    #output_csv_path = os.path.join(DATAROOT, 'dataset.csv')
    output_csv_path = DATAROOT / 'dataset.csv'
    #states_df.to_csv(output_csv_path, index=False)  # Optional: index=False to exclude row index


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
    
#linux
#python3 generate_dataset.py --dataroot 'data/sets/nuscenes' --version 'v1.0-mini' --split 'mini_train'






#windows 
#python3 C:\Users\sara_\Desktop\MasterThesis\pgeon-main\example\nuscenes\generate_dataset.py --dataroot 'data/sets/nuscenes' --version 'v1.0-mini' --split 'mini_train'

