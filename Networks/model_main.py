# from bagpy import bagreader
# import bagpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams["figure.figsize"] = (15,7)
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os

import tensorflow as tf
import google.protobuf

import pandas as pd
import sys
import importlib


import rosbag
from sensor_msgs.msg import Image
import cv2

sys.path.insert(0, '/home/kats/Documents/Repos/aru-core/build/lib')
import aru_py_mesh

sys.path.append('../Modular files/')
import ImagePointSelector
import ImgFeatureExtactorModule
import DatasetHandler


Handler = DatasetHandler.Dataset_Handler_Depth_Data('/media/kats/DocumentData/Data/data_depth_selection/depth_selection/')
depth_est = aru_py_mesh.PyDepth("/home/kats/Documents/Repos/aru-core/src/mesh/config/mesh_depth.yaml")

import dataLoader
import Utils
import test_model



# TODO: Add masking to layers



cmap_gt = r"/home/kats/Documents/My Documents/Datasets/data_depth_selection/depth_selection/val_selection_cropped/dense_groundtruth"
data_dir = r"/home/kats/Documents/My Documents/Datasets/data_depth_selection/depth_selection/val_selection_cropped"
dense_gt = os.path.join(data_dir, "dense_groundtruth")
mask_dir = os.path.join(data_dir, "depth_masks")


depth = Handler.first_depth
depth_c = depth[100:, :]
cv2.imshow("depth", 255*(depth_c==0).astype("uint8"))
#
cv2.waitKey(0)
cv2.destroyWindow("depth")



model = tf.keras.models.




# Layers = 4
# HEIGHT = int(full_size[0]/scale_factor)
# WIDTH = int(full_size[1]/scale_factor)
# WIDTH = 14*(2**Layers)
# HEIGHT = 4*(2**Layers)
# print(HEIGHT, WIDTH)
#
#
# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=LR,
#     amsgrad=False,
# )
# model = DepthEstimationModel()
# # Define the loss function
# cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction="none"
# )
#
# # Compile the model
# model.compile(optimizer, loss=cross_entropy)
#
# train_loader = DataGenerator(
#     data=df[:260].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
# )
# validation_loader = DataGenerator(
#     data=df[260:].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
# )
