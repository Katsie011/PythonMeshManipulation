r"""
Implementation of High Resolution Tunable Reconstruction from https://arxiv.org/pdf/1511.00758.pdf
Implemented by:                                                         Michael Katsoulis March 2022


This is the pipeline implementation after experimenting with the concept in a jupyter notebook.

"""

# -----------------------------------------------------------------------------------------------------------------
#       Importing libraries
# -----------------------------------------------------------------------------------------------------------------
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
from scipy.spatial.distance import hamming
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull
import time
from matplotlib import cm

# import aru_py_mesh
import sys

sys.path.insert(0, "/home/kats/Documents/My Documents/UCT/Masters/Code/Modular files")
# import DatasetHandler
import ImgFeatureExtactorModule as ft
import TriangleUtilityFunctions

# import Kitti_Dataset_Files_Handler
import pykitti

# /home/kats/Documents/My Documents/Datasets/KITTI_cvlibs/2011_09_26/2011_09_26_drive_0001_sync
base = r"/home/kats/Documents/My Documents/Datasets/KITTI_cvlibs/"
date = "2011_09_26"
drive = "0001"

data = pykitti.raw(base, date, drive)

from Functions_TunableReconstruction import *

# fig, ax = plt.subplots(1,2)
# ax[0].imshow(np.array(data.get_cam0(0)))
# ax[1].imshow(np.array(data.get_cam1(0)))
#
# ax[0].set_title("Test Left Img")
# ax[1].set_title("Test Right Img")
# plt.show()

# -----------------------------------------------------------------------------------------------------------------
#       Declaring Hyperparameters
# -----------------------------------------------------------------------------------------------------------------

NUM_INITIAL_FEATURES = 50
LOWE_DISTANCE_RATIO = 0.8
MAX_DISTANCE = 100  # meters
MAX_NUM_FEATURES_DETECT = 500
MIN_DISTANCE = 2
TRIANGLE_SAMPLES_PER_PIX_SQUARED = 1 / 10 ** 2
DETECTOR = 'ORB'
INTERPOLATING_POINTS = 2000

# -----------------------------------------------------------------------------------------------------------------
#       Extracting Features
# -----------------------------------------------------------------------------------------------------------------

img0 = np.array(data.get_cam0(0))
img1 = np.array(data.get_cam1(0))

if DETECTOR == 'ORB':
    det = ft.FeatureDetector(det_type='orb', max_num_ft=1000)
else:
    det = ft.FeatureDetector(det_type='sift', max_num_ft=MAX_NUM_FEATURES_DETECT)
# kp0 = det.detect(img0)
# kp1 = det.detect(img1)

kp0, des0 = det.detector.detectAndCompute(img0, None)  # left
kp1, des1 = det.detector.detectAndCompute(img1, None)  # right

kp0 = np.array(kp0);
kp1 = np.array(kp1)

# BFMatcher with default params
bf = cv2.BFMatcher(normType=cv2.NORM_L2)
matches = bf.knnMatch(des1, des0, k=2)  # img0 is train, img1 is query. Looking for right features in left

# Apply ratio test to select good features
good = []
for m, n in matches:
    if m.distance < LOWE_DISTANCE_RATIO * n.distance:
        good.append([m])

# getting the (u,v) points for the good matches
match_idxs = np.zeros((len(good), 2), dtype=int)
for i, m in enumerate(good):
    match_idxs[i] = [m[0].trainIdx, m[0].queryIdx]

# -----------------------------------------------------------------------------------------------------------------
#       Stereo Depth
# -----------------------------------------------------------------------------------------------------------------
K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(data.calib.P_rect_10)
t = t[:3] / t[3]  # normalising translation vector
t.squeeze()

# Getting disparity:
left_uv = keyPoint_to_UV(kp0[match_idxs[:, 0]])
right_uv = keyPoint_to_UV(kp1[match_idxs[:, 1]])
d = np.linalg.norm(left_uv - right_uv, axis=1)
depth = disparity_to_depth(d, K, t)

ft_uvd = np.hstack((left_uv, depth.reshape((-1, 1))))
ft_uvd = ft_uvd[ft_uvd[:, 2] < MAX_DISTANCE]  # Validating points

# Building mesh
d_mesh = Delaunay(ft_uvd[:, :2])

δ = cv2.Laplacian(img0, cv2.CV_64F)
δ = np.abs(δ)

idx = np.argpartition(δ.flatten(), -INTERPOLATING_POINTS)[-INTERPOLATING_POINTS:]
gradient_pts = np.unravel_index(idx, img0.shape)

interpolated_uv = np.stack((gradient_pts[1], gradient_pts[0]), axis=-1)
# mesh_pts_3d = ft_uvd.copy()

interpolated_pts = barycentric_interpolation(d_mesh, ft_uvd, interpolated_uv)



# -----------------------------------------------------------------------------------------------------------------
#       Cost Calculation
# -----------------------------------------------------------------------------------------------------------------

new_census = np.abs(get_disparity_census_cost(interpolated_pts, img0, img1, K, t, num_disparity_levels=5))

# MAX_COST = 0.8  # Hardcoded values
# MIN_COST = 0.2  # Hardcoded values

MAX_COST = new_census[:, new_census.shape[1] // 2].mean()  # Using shifting boundaries
MIN_COST = new_census.min(axis=1).mean() + new_census.min(axis=1).std()  # Statistically defined minimum
FT_COSTS = np.abs(get_disparity_census_cost(ft_uvd, img0, img1, K, t, num_disparity_levels=1)).mean()
print("-----------------------------")
print("Cost Calculation:")

print(f"Mean {new_census.mean()}")
print(f"Max {new_census.max()}")
print(f"Min {new_census.min()}")

print(f"Setting MAX_COST to {MAX_COST}")
print(f"Setting MIN_COST to {MIN_COST}")
print(f"Average cost from features for mesh construction was {FT_COSTS}")

min_cost_idx = new_census.min(axis=1) < MIN_COST

# Using the best match in the window at this point. It might not be right
max_cost_idx = new_census.min(axis=1) > MAX_COST

# C_g = np.nan*np.ones((len(new_census), 4)) # (u,v,d,c)
C_g = -1 * np.ones((len(new_census), 4))  # (u,v,d,c)
C_b = -1 * np.ones((len(new_census), 4))  # (u,v,d, c)

# good points are those with the lowest cost
C_g[min_cost_idx] = np.hstack((interpolated_pts[min_cost_idx, :2],
                               depth_to_disparity(interpolated_pts[min_cost_idx, 2], K, t).reshape((-1, 1)),
                               new_census.min(axis=1)[min_cost_idx].reshape((-1, 1))))
C_g_d_arg = np.argmin(new_census, axis=1)
C_b_d_arg = np.argmax(new_census, axis=1)

# Using the best match (new_census.min)
C_b[max_cost_idx] = np.hstack((interpolated_pts[max_cost_idx, :2],
                               depth_to_disparity(interpolated_pts[max_cost_idx, 2], K, t).reshape((-1, 1)),
                               new_census.min(axis=1)[max_cost_idx].reshape((-1, 1))))


print(f"Total num census cost pts {len(new_census)}")
print(f"Number of good pts to resample {(C_g.sum(axis=1) !=-3).sum()}")
print(f"Number of bad pts to resample {(C_b.sum(axis=1) !=-3).sum()}")


# -----------------------------------------------------------------------------------------------------------------
#       Resampling
# -----------------------------------------------------------------------------------------------------------------

# Need to implement selection of high cost points for the occupancy grids
# Need to implement selection of high cost points for the occupancy grids
# Need to implement selection of high cost points for the occupancy grids
# Need to implement selection of high cost points for the occupancy grids
# Need to implement selection of high cost points for the occupancy grids
# Need to implement selection of high cost points for the occupancy grids
# Need to implement selection of high cost points for the occupancy grids


# Epipolar search
F, mask = cv2.findFundamentalMat(left_uv.astype(np.int32),right_uv.astype(np.int32),cv2.FM_LMEDS) # getting the fundamental matrix
# Probably need to filter the matches used for this to only good points

# Need to search along the epipolar lines






