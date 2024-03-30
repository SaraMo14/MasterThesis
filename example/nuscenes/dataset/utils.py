import numpy as np

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
    #current_yaw = quaternion_yaw(current_rotation)
    #prev_yaw = quaternion_yaw(prev_rotation)

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
