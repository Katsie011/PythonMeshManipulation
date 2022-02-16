# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import google.protobuf
from bagpy import bagreader
import bagpy
import numpy as np

import pandas as pd
import sys

import rosbag
from sensor_msgs.msg import Image

# sys.path.insert(0, '/home/paulamayo/code/aru-core/build/lib/')
# sys.path.insert(0, '/home/kats/Documents/Repos/aru-core/build/lib')
# sys.path.insert(0, '/home/kats/DocumRents/Repos/aru-core/src/cmake-build-debug/lib')
sys.path.insert(0, '/home/kats/Documents/Repos/aru-core/src/cmake-build-debug/lib')
import cv2
import aru_py_mesh

def get_error_map(dense_map_1, dense_map_2):
    error_map = dense_map_1 - dense_map_2
    return error_map.copy()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sparse_depth = cv2.imread('/media/kats/DocumentData/Data/data_depth_selection/depth_selection/val_selection_cropped'
                              '/groundtruth_depth/2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png')
    img_left = cv2.imread('/media/kats/DocumentData/Data/data_depth_selection/depth_selection/val_selection_cropped/'+
                          'image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
    depth_est = aru_py_mesh.PyDepth("/home/kats/Documents/Repos/aru-core/src/mesh/config/mesh_depth.yaml")

    sparse_depth2 = cv2.imread('/media/kats/DocumentData/Data/data_depth_selection/depth_selection/val_selection_cropped'
                              '/groundtruth_depth/2011_09_26_drive_0002_sync_groundtruth_depth_0000000008_image_03.png')
    img_left_2 = cv2.imread('/media/kats/DocumentData/Data/data_depth_selection/depth_selection/val_selection_cropped/' +
                          'image/2011_09_26_drive_0002_sync_image_0000000005_image_03.png')


    sparse_depth = cv2.cvtColor(sparse_depth, cv2.COLOR_BGR2GRAY)
    sparse_depth = np.single(sparse_depth)

    sparse_depth2 = cv2.cvtColor(sparse_depth2, cv2.COLOR_BGR2GRAY)
    sparse_depth2 = np.single(sparse_depth2)

    print("Sparse depth of type:", sparse_depth.dtype)
    print(f"Shapes of inputs are:\n\tImg - {img_left.shape}\n\tDepth - {sparse_depth.shape}")

    # Make sure sparse_depth is float
    dense_depth = depth_est.create_dense_depth(sparse_depth)
    dense_depth2 = depth_est.create_dense_depth(sparse_depth2)

    print("Size of dense depth: ", dense_depth.shape)

    colour_sparse = cv2.applyColorMap
    cv2.imshow("Sparse", sparse_depth)
    cv2.waitKey(0)

    cv2.imshow("Error", get_error_map(dense_depth, dense_depth2))
    cv2.waitKey(0)


