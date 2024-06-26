import numpy as np
from nuscenes.prediction import convert_global_coords_to_local
import pandas as pd

def velocity(current_translation, prev_translation, time_diff: float) -> float:
    """
    Function to compute velocity between ego vehicle positions.
    
    :param current_translation: Translation [x, y, z] for the current timestamp.
    :param prev_translation: Translation [x, y, z] for the previous timestamp.
    :param time_diff: How much time has elapsed between the records.

    Return:
        velocity: The function ignores the Z component (diff[:2] limits the difference to the X and Y components).
    """
    if time_diff == 0:
        return np.NaN
    diff = (np.array(current_translation) - np.array(prev_translation)) / time_diff
    return np.linalg.norm(diff[:2])

def quaternion_yaw(q):
    """
    Calculate the yaw from a quaternion.
    :param q: Quaternion [w, x, y, z]
    """
    return np.arctan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))


def heading_change_rate(current_yaw, prev_yaw, time_diff: float) -> float:
    """
    Function to compute the rate of heading change.
    """
    if time_diff == 0:
        return np.NaN

    return (current_yaw- prev_yaw) / time_diff

def acceleration(current_velocity, prev_velocity,
                 time_diff: float) -> float:
    """
    Function to compute acceleration between sample annotations.
    :param current_velocity: Current velocity.
    :param prev_velocity: Previous velocity.
    :param time_diff: How much time has elapsed between the records.
    """
    if time_diff == 0:
        return np.NaN
    return (current_velocity - prev_velocity) / time_diff


def convert_coordinates( group: pd.DataFrame) -> pd.DataFrame:
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
        group['yaw_rate'] = group['yaw'].diff() / valid_time_diffs

        return group


def train_test_split_by_scene(df, test_size=0.2, random_state=42):
    """
    Split DataFrame into train and test sets based on scene tokens.

    Parameters:
        df (pd.DataFrame): DataFrame to be split.
        test_size (float): Proportion of the dataset to include in the test split (0.0 to 1.0).
        random_state (int): Seed for random number generation.

    Returns:
        train_df (pd.DataFrame): Train set.
        test_df (pd.DataFrame): Test set.
    """
    # Get unique scene tokens
    scene_tokens = df['scene_token'].unique()

    # Randomly shuffle the scene tokens
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(scene_tokens)

    # Calculate the number of tokens for the test set
    num_tokens_test = int(len(scene_tokens) * test_size)

    # Split scene tokens into train and test sets
    test_tokens = scene_tokens[:num_tokens_test]
    train_tokens = scene_tokens[num_tokens_test:]

    # Filter DataFrame based on train and test tokens
    train_df = df[df['scene_token'].isin(train_tokens)]
    test_df = df[df['scene_token'].isin(test_tokens)]

    return train_df, test_df
