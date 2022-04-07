"""
This file aims to provide an easy interface with the data collected using the Husky robot.

Features that need to be implemented:

    - Bagfile to Dataset function
        - Argument parsing to select data

    - Data generators
        - Create generators to access data in the subfolders
    - Calibration file access

    bagfile extractor based off of bagfile to image script by Matt Church 2021
"""
import os
import dateutil.parser as dparser
import rosbag
import numpy as np
import pandas as pd
import cv2
import tqdm
import sys

sys.path.insert(0, '/opt/ros/melodic/lib/python2.7/dist-packages/')
import ros_numpy as rnp


class Dataset_Handler():

    def __init__(self, data_path, low_memory=True):
        self.data_path = data_path
        self.low_memory = low_memory

        self.left_image_path = os.path.join(self.data_path, "image_00/data")
        self.right_image_path = os.path.join(self.data_path, "image_01/data")
        self.lidar_path = os.path.join(self.data_path, "velodyne_points/data")

        self.left_image_files = []
        self.right_image_files = []
        self.lidar_files = []

        # Get names of files to iterate through
        if os.path.exists(self.left_image_path):
            self.left_image_files = os.listdir(self.left_image_path)

        if os.path.exists(self.right_image_path):
            self.right_image_files = os.listdir(self.right_image_path)

        if os.path.exists(self.lidar_path):
            self.lidar_files = os.listdir(self.lidar_path)

        self.num_frames = len(self.left_image_files)
        self.times = np.array(pd.read_csv(os.path.join(data_path, 'image_00/timestamps.txt'), delimiter='\n'))

        """
        # Calibration files
        # TODO
        self.calib_path = 
        self.calib.K_cam0 = 
        self.calib.K_cam1 = 
        self.calib.T_cam0_cam1 = 
        self.calib.T_cam0_velo = 
        self.calib.R_rect_00 =
        self.calib.P_rect_00 = 
        """


        """
        NOTES:
        
        - Still need to have a way of reading in calibration files for camera-to-camera calibration
            Also need to have lidar-to-camera0 calibration files. 
        
        - This could probably benefit from having generators built in. 
            Would allow iterating over images in loops. 
            You're welcome to add that in as a new function if you would like
        """

    def get_cam0(self, frame_num):
        r"""Returns the left image at the given frame"""
        assert frame_num <= self.num_frames, "Frame number greater than number of files"
        assert len(self.left_image_files) > 0, f"No image files found at {self.left_image_path}"

        im = os.path.join(self.left_image_path, self.left_image_files[frame_num])
        return cv2.imread(im)

    def get_cam1(self, frame):
        r"""Returns the right image at the given frame"""
        assert frame <= self.num_frames, "Frame number greater than number of files"
        assert len(self.right_image_files) > 0, f"No image files found at {self.right_image_path}"

        im = os.path.join(self.right_image_path, self.right_image_files[frame])
        return cv2.imread(im)

    def get_lidar(self, frame):
        r"""Returns the lidar points at the given frame"""
        assert frame <= self.num_frames, "Frame number greater than number of files"
        assert len(self.lidar_files) > 0, f"No Lidar files found at {self.lidar_path}"
        file = os.path.join(self.lidar_path, self.lidar_files[frame])
        return np.load(file)

    def get_times(self):
        return self.times

    def get_num_frames(self):
        return self.num_frames


def bagfile_to_dataset(path_bagfile, path_output_dataset, camera_mat=None, dist_mat=None, custom_date=False,
                       images=True,
                       lidar=True, verbose=True, overwrite=False):
    # Topics:
    #     / velodyne_points
    #     "/camera/image_left/"
    #     "/camera/image_right/"

    if images:
        left_topic = "/camera/image_left/"
        right_topic = "/camera/image_right/"
    if lidar:
        lidar_topic = "/velodyne_points/"

    if os.path.isdir(path_bagfile):
        bags = []
        for f in os.listdir(path_bagfile):
            _, extension = os.path.splitext(f)
            if extension == '.bag':
                bags.append(os.path.join(path_bagfile, f))

    if verbose:
        print(f"Bags for extraction: \n{bags}")

    for bag in bags:
        if custom_date:
            date = custom_date
        else:
            _, bag_name = os.path.split(bag)
            date = dparser.parse(os.path.splitext(bag_name)[0], fuzzy=True).strftime("%Y_%m_%d_%H_%M")

        # Check if folders exist
        """
        Kitti Data Structure:
        _______________________________
        data_dir
            <date>
                <date>_drive_000#_sync
                    image_00
                        data
                            <images.png>
                        timestamps.txt
                    image_01
                        ...
                    image_02
                        ...
                    image_03
                        ...
                    velodyne_points
                        data
                            #.bin
                        timestamps.txt
                calib_cam_to_cam.txt
                calib_imu_to_velo.txt
                calib_velo_to_cam.txt
        """

        if verbose:
            print("--------------------------------------------------------------------------------")
            print(f"\t\tChecking and Making folders for {bag_name}")
            print("--------------------------------------------------------------------------------")

        date_folder = os.path.join(path_output_dataset, date)
        if not os.path.exists(date_folder):
            os.makedirs(date_folder)
            if verbose: print(f"\t-Master folder created: {date_folder}")
        else:
            print("Folder already exists for this date and time.")
            if overwrite:
                print("\tOverwriting")
            else:
                print("\tSkipping")
                continue

        if not os.path.exists(os.path.join(date_folder, "calib.txt")):
            if verbose: print("\n\t- Making image calibration file.")
            if camera_mat is None:
                print("\t\tNo camera matrix supplied. Defaulting to preset.")
                camera_mat = np.array([[531.14774, 0.000000, 637.87114],
                                       [0.000000, 531.26312, 331.27469],
                                       [0.000000, 0.000000, 1.000000]])
            if dist_mat is None:
                print("\t\tNo distortion matrix supplied. Defaulting to preset.")
                dist_mat = np.array([-0.048045, 0.010493, -0.000281, -0.001232, 0.000000])

            file = open(os.path.join(date_folder, "calib.txt"), 'w')
            file.write(f"K={camera_mat}\n")
            file.write(f"dist={dist_mat}\n")
            file.close()
            if verbose: print("\t...... Done\n")

        if images:
            left_im_path = os.path.join(path_output_dataset, date, 'image_00', 'data')
            right_im_path = os.path.join(path_output_dataset, date, 'image_01', 'data')
            if not os.path.exists(left_im_path):
                # left camera dir does not exist
                os.makedirs(left_im_path)
                if verbose: print("\t-Left image data folder created")
            if not os.path.exists(right_im_path):
                # right camera dir does not exist
                os.makedirs(right_im_path)
                if verbose: print("\t-Right image data folder created")

        if lidar:
            velodyne_path = os.path.join(path_output_dataset, date, 'velodyne_points', 'data')
            if not os.path.exists(velodyne_path):
                os.makedirs(velodyne_path)
                if verbose: print("\t-Velodyne data folder created")

        if verbose: print("...Folders created")
        if verbose: print("_________________________________\n")

        """ 
        --------------------------------------------------------------------------------
          Extracting
        --------------------------------------------------------------------------------
        
        Now ready for extracting data
            
            - Will do images and/or lidar based on what is needed
            
            Adding progress bars for visualisation
            have to use the generator objects returned by rosbag for this to work. 
            Not ideal but shouldn't slow down much.

        """
        if verbose:
            print("--------------------------------------------------------------------------------")
            print(f"\t\tExtracting Data from {bag_name}")
            print("--------------------------------------------------------------------------------")

        if verbose: print("\tCreating rosbag variable. This may take a while")
        ros_bag = rosbag.Bag(bag)
        print("\tSTILL NEED TO CHECK IF THE TIMESTAMPS MATCH FOR IMAGESe")

        if images:
            if verbose: print("\t- Extracting Images")
            if verbose: print("\t\t- Extracting Left Images:")
            len_left = ros_bag.get_message_count(left_topic)
            len_right = ros_bag.get_message_count(right_topic)
            if (len_left == len_right): print("WARNING:Number of images on left and right cameras do not match")

            left_gen = ros_bag.read_messages(topics=left_topic)

            # Making the timestamp files:
            timestamp_path, _ = os.path.split(left_im_path)
            left_timestamps = open(os.path.join(timestamp_path, "timestamps.txt"), 'w')
            for l in tqdm.trange(len_left):
                # for topic, msg, t in ros_bag.read_messages(topics=left_topic):
                topic, msg, t = next(left_gen)
                left_timestamps.write(str(t) + '\n')
                bag_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if camera_mat is not None:
                    if dist_mat is None:
                        print("Please provide distortion coefficients")
                        break
                    else:
                        out_img = cv2.undistort(bag_img, camera_mat, dist_mat)
                else:
                    out_img = bag_img
                cv2.imwrite(os.path.join(left_im_path, str(t) + ".png"), out_img)

            # Remember to close files
            left_timestamps.close()

            if verbose: print("\t\t- Extracting Right Images:")
            right_gen = ros_bag.read_messages(topics=right_topic)

            print()
            print("NOTE: Need to clean the rest of right extraction.\n-- assert if timestamps are approx equal?")
            print()

            # Timestamp writer
            timestamp_path, _ = os.path.split(right_im_path)
            right_timestamps = open(os.path.join(timestamp_path, "timestamps.txt"), 'w')

            for right_index in tqdm.trange(len_right):
                # for topic, msg, t in ros_bag.read_messages(topics=topic_right):
                topic, msg, t = next(right_gen)

                right_timestamps.write(str(t) + '\n')

                bag_img = np.frombuffer(msg.data,
                                        dtype=np.uint8).reshape(msg.height,
                                                                msg.width,
                                                                -1)
                if camera_mat is not None:
                    if dist_mat is None:
                        print("Please provide distortion coefficients")
                        break
                    else:
                        out_img = cv2.undistort(bag_img, camera_mat, dist_mat)
                else:
                    out_img = bag_img
                cv2.imwrite(os.path.join(right_im_path, str(t) + ".png"), out_img)

            right_timestamps.close()

            if verbose: print("\n\t......Done extracting images\n")

        if lidar:
            if verbose: print("\n\t- Extracting Lidar Data")
            len_lidar = ros_bag.get_message_count(lidar_topic)
            for lidar_index in tqdm.trange(len_lidar):
                lidar_gen = ros_bag.read_messages(topics=lidar_topic)
                topic, msg, t = next(lidar_gen)
                pts = rnp.point_cloud2.pointcloud2_to_xyz_array(msg)

                lidar_file = os.path.join(velodyne_path, f"{t}.npy")
                np.save(lidar_file, pts, allow_pickle=False)

        if verbose:
            print("\t....Done extracting lidar data")
            print("\n\n")
            print("------------------------------------------")
            print(f"Summary for {bag_name}:")
            print("------------------------------------------")
            if images:
                print(f"\t- Extracted: {len_left} left images")
                print(f"\t- Extracted: {len_right} right images")
            if lidar:
                print(f"\t- Extracted: {len_lidar} lidar scans")

            print()
            print(f"\tFiles saved to {date_folder}")
            print()
            print("------------------------------------------")
            print()

    print("\n\n------------------------------------------")
    print("No other bagfiles found in folder.")
    print("Job is done.")
    print("Goodbye.")


if __name__ == "__main__":
    print("Running tests")

    print("Testing data class")
    data_path = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/test1/2022_02_23_09_00"
    print(f"Looking for dataset in:{data_path}")
    dataset = Dataset_Handler(data_path=data_path)

    import matplotlib.pyplot as plt
    frame_list = [0, 10, 20, 100, dataset.num_frames - 10]
    for frame in frame_list:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(dataset.get_cam0(frame))
        ax[1].imshow(dataset.get_cam1(frame))

        ax[0].set_title("Left image")
        ax[1].set_title("Right image")

        ax[0].axis('off')
        ax[1].axis('off')

        fig.suptitle(f"Images for frame {frame}")

        plt.show()


    print("Still need to formulate tests for bagfile extraction")
