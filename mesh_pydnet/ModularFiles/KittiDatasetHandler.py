# Credit to Nate Cibik https://github.com/FoamoftheSea/KITTI_visual_odometry
# for the dataset handler

import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd



class Dataset_Handler():

    def __init__(self, sequence, lidar=True, progress_bar=True, low_memory=True):

        # This will tell our odometry function if handler contains lidar info
        self.lidar = lidar
        # This will tell odometry functin how to access data from this object
        self.low_memory = low_memory

        # Set file paths and get ground truth poses
#         self.seq_dir = '../dataset/sequences/{}/'.format(sequence)
#         self.poses_dir = '../dataset/poses/{}.txt'.format(sequence)
#         poses = pd.read_csv(self.poses_dir, delimiter=' ', header=None)

        # Get names of files to iterate through
        self.left_image_files = os.listdir(self.seq_dir + 'image_0')
        self.right_image_files = os.listdir(self.seq_dir + 'image_1')
        self.velodyne_files = os.listdir(self.seq_dir + 'velodyne')
        self.num_frames = len(self.left_image_files)
        self.lidar_path = self.seq_dir + 'velodyne/'

        # Get calibration details for scene
        # P0 and P1 are Grayscale cams, P2 and P3 are RGB cams
        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape((3,4))
        self.P1 = np.array(calib.loc['P1:']).reshape((3,4))
        self.P2 = np.array(calib.loc['P2:']).reshape((3,4))
        self.P3 = np.array(calib.loc['P3:']).reshape((3,4))
        # This is the transformation matrix for LIDAR
        self.Tr = np.array(calib.loc['Tr:']).reshape((3,4))

        # Get times and ground truth poses
        self.times = np.array(pd.read_csv(self.seq_dir + 'times.txt',
                                          delimiter=' ',
                                          header=None))
        self.gt = np.zeros((len(poses), 3, 4))
        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

        # Get images and lidar loaded
        if self.low_memory:
            # Will use generators to provide data sequentially to save RAM
            # Use class method to set up generators
            self.reset_frames()
            # Store original frame to memory for testing functions
            self.first_image_left = cv2.imread(self.seq_dir + 'image_0/'
                                               + self.left_image_files[0], 0)
            self.first_image_right = cv2.imread(self.seq_dir + 'image_1/'
                                               + self.right_image_files[0], 0)
            self.second_image_left = cv2.imread(self.seq_dir + 'image_0/'
                                               + self.left_image_files[1], 0)
            if self.lidar:
                self.first_pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[0],
                                                    dtype=np.float32,
                                                    count=-1).reshape((-1, 4))
            self.imheight = self.first_image_left.shape[0]
            self.imwidth = self.first_image_left.shape[1]

        else:
            # If RAM is not a concern (>32GB), pass low_memory=False
            if progress_bar:
                import progressbar
                bar = progressbar.ProgressBar(max_value=self.num_frames)
            self.left_image = []
            self.images_right = []
            self.pointclouds = []
            for i, name_left in enumerate(self.left_image_files):
                name_right = self.right_image_files[i]
                self.left_image.append(cv2.imread(self.seq_dir + 'image_0/' + name_left))
                self.images_right.append(cv2.imread(self.seq_dir + 'image_1/' + name_right))
                if self.lidar:
                    pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[i],
                                             dtype=np.float32,
                                             count=-1).reshape([-1,4])
                    self.pointclouds.append(pointcloud)
                if progress_bar:
                    bar.update(i+1)

            self.imheight = self.left_image[0].shape[0]
            self.imwidth = self.left_image[0].shape[1]
            # Keep consistent instance variable names as when using low_memory
            self.first_image_left = self.left_image[0]
            self.first_image_right = self.images_right[0]
            self.second_image_left = self.left_image[1]
            if self.lidar:
                self.first_pointcloud = self.pointclouds[0]

    def reset_frames(self):
        # Resets all generators to the first frame of the sequence
        self.left_image = (cv2.imread(self.seq_dir + 'image_0/' + name_left, 0)
                            for name_left in self.left_image_files)
        self.images_right = (cv2.imread(self.seq_dir + 'image_1/' + name_right, 0)
                            for name_right in self.right_image_files)
        if self.lidar:
            self.pointclouds = (np.fromfile(self.lidar_path + velodyne_file,
                                            dtype=np.float32,
                                            count=-1).reshape((-1, 4))
                                for velodyne_file in self.velodyne_files)
        pass


    
    
class Dataset_Handler_Depth_Data():
    def __init__(self, root, lidar=True, progress_bar=True, low_memory=True, verbose = False):

        # This will tell our odometry function if handler contains lidar info
        self.lidar = lidar
        # This will tell odometry functin how to access data from this object
        self.low_memory = low_memory
        # Setting logging
        self.verbose = verbose


        # Set file paths and get ground truth poses
        self.val_dir = '{}/val_selection_cropped/'.format(root)
        self.poses_dir = '{}/test_depth_completion_anonymous/'.format(root)

        # Get names of files to iterate through
        self.left_image_files = os.listdir(self.val_dir + 'image')
        self.instrinsics = os.listdir(self.val_dir + 'intrinsics')
        self.velodyne_files = os.listdir(self.val_dir + 'velodyne_raw')
        self.depth_files = os.listdir(self.val_dir + 'groundtruth_depth')
        self.num_frames = len(self.left_image_files)

        if self.verbose:
            print(self.num_frames, " Frames read in")

        # Get images and lidar loaded
        if self.low_memory:
            if self.verbose:
                print("Low memory mode")
                print("Setting up first frames")
            # Will use generators to provide data sequentially to save RAM
            # Use class method to set up generators
            self.reset_frames()
            # Store original frame to memory for testing functions
            self.first_image_left = cv2.imread(self.val_dir + 'image/'
                                               + self.left_image_files[0], 0)
            self.first_depth = cv2.imread(self.val_dir + 'groundtruth_depth/'
                                               + self.depth_files[0], 0)
            self.second_image_left = cv2.imread(self.val_dir + 'image/'
                                               + self.left_image_files[1], 0)
            if self.lidar:
                self.first_lidar_img = cv2.imread(self.val_dir + 'velodyne_raw/' +
                                                   self.velodyne_files[0], 0)
            self.imheight = self.first_image_left.shape[0]
            self.imwidth = self.first_image_left.shape[1]

        else:
            # If RAM is not a concern (>32GB), pass low_memory=False
            if progress_bar:
                import progressbar
                bar = progressbar.ProgressBar(max_value=self.num_frames)
            self.left_image = []
            self.depth = []
            for i, name_left in enumerate(self.left_image_files):
                self.left_image.append(cv2.imread(self.val_dir + 'image/' + name_left, 0))
                self.depth.append(cv2.imread(self.val_dir + 'groundtruth_depth/' +
                           self.depth_files[i], 0))
                if self.lidar:
                    pt_img =  cv2.imread(self.val_dir + 'velodyne_raw/' +
                                                   self.velodyne_files[i], 0)
                    self.pointclouds.append(pt_img)
                if progress_bar:
                    bar.update(i+1)

            self.imheight = self.left_image[0].shape[0]
            self.imwidth = self.left_image[0].shape[1]
            # Keep consistent instance variable names as when using low_memory
            self.first_image_left = self.left_image[0]
            self.second_image_left = self.left_image[1]
            if self.lidar:
                self.first_depth = self.depth[0]

    def reset_frames(self):
        # Resets all generators to the first frame of the sequence
        if self.verbose:
            print("Re-initialising generators")
        self.left_image = (cv2.imread(self.val_dir + 'image/' + name_left, 0)
                            for name_left in self.left_image_files)
        self.depth = (cv2.imread(self.val_dir + 'groundtruth_depth/' + name_depth, 0)
                            for name_depth in self.depth_files)
        if self.lidar:
            self.pointclouds = (cv2.imread(self.val_dir + 'velodyne_raw/' + velodyne_file, 0)
                                for velodyne_file in self.velodyne_files)
        pass
    
    
    
    
    
    

if __name__ == "__main__":
    Dataset_Handler_Depth_Data('/media/kats/DocumentData/Data/data_depth_selection/depth_selection/')
