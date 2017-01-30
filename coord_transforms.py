import numpy as np


def rotate_3d(pt, yaw, pitch, roll):
    """rotate a 3D point in ENU coordinates by the given yaw, pitch, and roll.
       pt: 3-element tuple of E, N, and U coordinates
       yaw: yaw angle in degrees [-180, 180], positive right, negative left
       pitch: pitch angle in degrees [-90, 90], positive up, negative down
       roll: roll angle in degrees [-180, 180], positive clockwise, negative counter-clockwise
       return: 3-element tuple of rotated coordinates"""

    # yaw: rotate around Z (up) axis
    # pitch: rotate around X (east) axis
    # roll: rotate around Y (north) axis

    pt = np.array(pt)

    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)

    rot_yaw = np.array([[np.cos(yaw), np.sin(yaw), 0],
                       [-np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
    rot_pitch = np.array([[1, 0, 0],
                         [0, np.cos(pitch), -np.sin(pitch)],
                         [0, np.sin(pitch), np.cos(pitch)]])
    rot_roll = np.array([[np.cos(roll), 0, np.sin(roll)],
                        [0, 1, 0],
                        [-np.sin(roll), 0, np.cos(roll)]])
    
    rot = np.dot(np.dot(rot_yaw, rot_pitch), rot_roll)
    return tuple(np.dot(pt, rot))
