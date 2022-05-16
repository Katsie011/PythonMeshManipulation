import numpy as np
import cv2
# import pykitti
#
# print("Loading Kitti")
# base = r"/home/kats/Datasets/KITTI_cvlibs/"
# date = "2011_09_26"
# drive = "0001"
# data = pykitti.raw(base, date, drive)
# print("Loaded")

import sys

sys.path.insert(0, '/home/kats/Documents/Repos/aru-core/build/lib/')
import aru_py_mesh

# global NUM_INITIAL_FEATURES
# global LOWE_DISTANCE_RATIO
# global MAX_DISTANCE
# global MAX_NUM_FEATURES_DETECT
# global MIN_DISTANCE
# global TRIANGLE_SAMPLES_PER_PIX_SQUARED
# global DETECTOR
# global INTERPOLATING_POINTS

global K
global R
global t

# K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(data.calib.P_rect_10)
# t = t[:3] / t[3]  # normalising translation vector
# t.squeeze()

NUM_INITIAL_FEATURES = 200
LOWE_DISTANCE_RATIO = 0.8
MAX_DISTANCE = 100  # meters  This is 4px disparity
MAX_NUM_FEATURES_DETECT = 500
MIN_DISTANCE = 2
TRIANGLE_SAMPLES_PER_PIX_SQUARED = 1 / 10 ** 2
# DETECTOR = 'ORB'
DETECTOR = 'SIFT'
INTERPOLATING_POINTS = 200
NUM_SUPPORT_PTS_PER_OCCUPANCY_GRID = 2
RESAMPLING_ITERATIONS = 3

IMAGE_SHAPE = (720, 1280, 3)
PREDICTION_SHAPE = (384, 640)

import PythonMeshManipulation.mesh_pydnet.HuskyCalib as HuskyCalib
ZED_t = HuskyCalib.t_cam0_velo
ZED_R = HuskyCalib.R_rect_cam0

t = ZED_t
K = HuskyCalib.left_camera_matrix
R = ZED_R
