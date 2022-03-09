import numpy as np
import pykitti
import cv2

base = r"/home/kats/Documents/My Documents/Datasets/KITTI_cvlibs/"
date = "2011_09_26"
drive = "0001"
data = pykitti.raw(base, date, drive)
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

K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(data.calib.P_rect_10)


NUM_INITIAL_FEATURES = 200
LOWE_DISTANCE_RATIO = 0.8
MAX_DISTANCE = 100  # meters
MAX_NUM_FEATURES_DETECT = 500
MIN_DISTANCE = 2
TRIANGLE_SAMPLES_PER_PIX_SQUARED = 1 / 10 ** 2
# DETECTOR = 'ORB'
DETECTOR = 'SIFT'
INTERPOLATING_POINTS = 1000
NUM_SUPPORT_PTS_PER_OCCUPANCY_GRID = 3






