import numpy as np
import yaml
import os


with open(r"/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/aru-calibration/ZED/left.yaml")\
        as file:
    left_vars = yaml.load(file, Loader=yaml.FullLoader)
with open(
        r'/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/aru-calibration/ZED/right.yaml')\
        as file:
    right_vars = yaml.load(file, Loader=yaml.FullLoader)

left_distortion_coefficients = np.array(left_vars['distortion_coefficients']['data']).reshape(
    (left_vars['distortion_coefficients']['rows'], left_vars['distortion_coefficients']['cols']))
left_rectification_matrix = np.array(left_vars['rectification_matrix']['data']).reshape(
    (left_vars['rectification_matrix']['rows'], left_vars['rectification_matrix']['cols']))
left_projection_matrix = np.array(left_vars['projection_matrix']['data']).reshape(
    (left_vars['projection_matrix']['rows'], left_vars['projection_matrix']['cols']))
left_camera_matrix = np.array(left_vars['camera_matrix']['data']).reshape(
    (left_vars['camera_matrix']['rows'], left_vars['camera_matrix']['cols']))

right_distortion_coefficients = np.array(right_vars['distortion_coefficients']['data']).reshape(
    (right_vars['distortion_coefficients']['rows'], right_vars['distortion_coefficients']['cols']))
right_rectification_matrix = np.array(right_vars['rectification_matrix']['data']).reshape(
    (right_vars['rectification_matrix']['rows'], right_vars['rectification_matrix']['cols']))
right_projection_matrix = np.array(right_vars['projection_matrix']['data']).reshape(
    (right_vars['projection_matrix']['rows'], right_vars['projection_matrix']['cols']))
right_camera_matrix = np.array(right_vars['camera_matrix']['data']).reshape(
    (right_vars['camera_matrix']['rows'], right_vars['camera_matrix']['cols']))


T_cam0_vel0 = np.array([[0.99996604, 0.007019, 0.0043191, 0.07807717],
                        [0.00419319, 0.01785629, -0.99983177, -0.2949953],
                        [-0.00709495, 0.99981593, 0.01782625, 0.00373244],
                        [0., 0., 0., 1.]])
R_rect_cam0 = np.array([[0.999966038877620, 0.00419318683524525, -0.00709494718561779],
                        [0.00701900482321433, 0.0178562859109133, 0.999815926370829],
                        [0.00431910438559592, -0.999831770968313, 0.0178262474927810]])

t_cam0_velo = np.array([0.0780771698321823, -0.294995301932968, 0.00373244344041508])
