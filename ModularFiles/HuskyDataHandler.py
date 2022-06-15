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
import yaml
from collections import namedtuple

"""
-------------------------------------------------------------------------
Hard coded calibration files
-------------------------------------------------------------------------

    This is not the right way to do it but modify these variables to point to the location of the aru-calibration 
    on your machine. 
    
    
    CHANGE THIS FOR YOUR MACHINE:
"""
########################################################################################################################
CAMERA_LEFT_CALIB_FILE = r'/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/aru-calibration/ZED/left.yaml'
CAMERA_RIGHT_CALIB_FILE = r'/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/aru-calibration/ZED/right.yaml'


########################################################################################################################


class DatasetHandler:

    def __init__(self, data_path, low_memory=True, time_sync=True, time_tolerance=0.1, timestamp_resolution=1e-9,
                 left_calib=CAMERA_LEFT_CALIB_FILE, right_calib=CAMERA_RIGHT_CALIB_FILE):
        self.data_path = data_path
        self.low_memory = low_memory

        self.left_image_path = os.path.join(self.data_path, "image_00/data")
        self.right_image_path = os.path.join(self.data_path, "image_01/data")
        self.lidar_path = os.path.join(self.data_path, "velodyne_points/data")

        self.has_lidar = os.path.exists(self.lidar_path)
        self.has_right_imgs = os.path.exists(self.right_image_path)

        self.time_tolerance = time_tolerance
        self.timestamp_resolution = timestamp_resolution

        self.left_image_files = []
        self.right_image_files = []
        self.lidar_files = []

        self.calib = load_calib(left_calib, right_calib)

        # Get names of files to iterate through
        assert os.path.exists(self.left_image_path), 'Error, left images not found'
        left_files = np.asarray(os.listdir(self.left_image_path), dtype=object)
        left_times = np.zeros(len(left_files))
        for i, f in enumerate(left_files):
            left_times[i] = int(os.path.splitext(f)[0])

        # Sorting to ensure that times are ascending.
        sort_ids = np.argsort(left_times)
        left_times = left_times[sort_ids]
        left_files = left_files[sort_ids]

        # Making an array to store matching right imgs and lidar imgs corresponding to each left.
        matches = -np.ones((len(left_files), 2), dtype=int)  # -1 if no match found

        # Load in the right files
        if self.has_right_imgs and time_sync:
            unsynced_right_files = np.asarray(os.listdir(self.right_image_path), dtype=object)
            unsynced_right_times = -1 * np.ones(len(unsynced_right_files))
            right_files = np.zeros(len(left_files), dtype=object)
            right_times = -1 * np.ones(len(right_files))
            for i, f in enumerate(unsynced_right_files):
                # Getting times
                unsynced_right_times[i] = int(os.path.splitext(f)[0])

            for i, lt in enumerate(left_times):
                # Need to get the first image where t_right > t_left
                # Right images are published after left but aquired simultaneously

                # getting the time difference w.r.t. current right time
                difference = unsynced_right_times - lt
                r_ind = np.argmin(np.abs(difference))  # minimum along left times

                if difference[r_ind] * timestamp_resolution < time_tolerance:
                    # getting the first frame where right is after left (first positive time)
                    right_files[i] = unsynced_right_files[r_ind]
                    right_times[i] = unsynced_right_times[r_ind]

                    # need to reassure that this matches all files and that right images are not published after left.

                    # removing the matched file from the right files that still need to be matched
                    # this is to avoid duplicate matches
                    ind = np.ones(len(unsynced_right_files), dtype=bool)
                    ind[r_ind] = False

                    unsynced_right_files = unsynced_right_files[ind]
                    unsynced_right_times = unsynced_right_times[ind]

                else:
                    print(f"Could not match {left_files[i]} with a right image")
                    # this is just for dev purposes

        if self.has_lidar and time_sync:
            unsynced_lidar_files = np.asarray(os.listdir(self.lidar_path), dtype=object)
            unsynced_lidar_times = -1 * np.ones(len(unsynced_lidar_files))
            lidar_files = np.zeros(len(left_files), dtype=object)
            lidar_times = -1 * np.ones(len(lidar_files))
            for i, f in enumerate(unsynced_lidar_files):
                # Getting times
                unsynced_lidar_times[i] = int(os.path.splitext(f)[0])

            for i, lt in enumerate(left_times):
                # Need to get the first image where t_right > t_left
                # Right images are published after left but aquired simultaneously

                # getting the time difference w.r.t. current right time
                difference = unsynced_lidar_times - lt
                lid_ind = np.argmin(np.abs(difference))  # minimum along left times

                if difference[lid_ind] * timestamp_resolution < time_tolerance:
                    # getting the first frame where right is after left (first positive time)
                    # l_ind = np.argmin(np.abs(difference), axis=0)  # minimum along left times
                    # deltas = left_times[l_ind] - unsynced_right_times  # Keep the time difference in an array
                    lidar_files[i] = unsynced_lidar_files[lid_ind]
                    lidar_times[i] = unsynced_lidar_times[lid_ind]

                    # need to reassure that this matches all files and that right images are not published after left.

                    # removing the matched file from the right files that still need to be matched
                    # this is to avoid duplicate matches

                    ind = np.ones(len(unsynced_lidar_files), dtype=bool)
                    ind[lid_ind] = False

                    unsynced_lidar_files = unsynced_lidar_files[ind]
                    unsynced_lidar_times = unsynced_lidar_times[ind]

                else:
                    print(f"Could not match {left_files[i]} with a lidar file")

        if time_sync:
            # Selecting for those images where there is a matching lidar and right image
            valid = ((right_times > 0) | (self.has_right_imgs == 0)) & ((lidar_times > 0) | (self.has_lidar == 0))

            # Trimming arrays and writing to object

            self.left_image_files = left_files[valid]
            self.left_times = left_times[valid]
            if os.path.exists(self.right_image_path):
                self.right_image_files = right_files[valid]
                self.right_times = right_times[valid]
            if os.path.exists(self.lidar_path):
                self.lidar_files = lidar_files[valid]
                self.lidar_times = lidar_times[valid]

        else:
            # Just returning all the files that are found
            self.left_files = np.asarray(os.listdir(self.left_image_path), dtype=object)
            self.left_times = np.zeros(len(self.left_files))
            for i, f in enumerate(self.left_files):
                self.left_times[i] = int(os.path.splitext(f)[0])

            if self.has_right_imgs:
                self.right_files = np.asarray(os.listdir(self.right_image_path), dtype=object)
                self.right_times = np.zeros(len(self.right_files))
                for i, f in enumerate(self.right_files):
                    self.right_times[i] = int(os.path.splitext(f)[0])

            if self.has_lidar:
                self.lidar_files = np.asarray(os.listdir(self.lidar_path), dtype=object)
                self.lidar_times = np.zeros(len(self.lidar_files))
                for i, f in enumerate(self.lidar_files):
                    self.lidar_times[i] = int(os.path.splitext(f)[0])

        # Setting the number of files
        self.num_frames = len(self.left_image_files)

        # making the variables for rectification.
        temp_im = cv2.imread(os.path.join(self.left_image_path, self.left_image_files[0]))
        self.img_shape = temp_im.shape
        # For distortion:
        self.left_mapx, self.left_mapy = cv2.initUndistortRectifyMap(self.calib.cam0_camera_matrix,
                                                                     self.calib.cam0_distortion_coefficients.squeeze(),
                                                                     None,
                                                                     self.calib.cam0_projection_matrix,
                                                                     self.calib.cam0_img_shape[::-1]
                                                                     , 5)

        self.right_mapx, self.right_mapy = cv2.initUndistortRectifyMap(self.calib.cam1_camera_matrix,
                                                                       self.calib.cam1_distortion_coefficients.squeeze(),
                                                                       None,
                                                                       self.calib.cam1_projection_matrix,
                                                                       self.calib.cam1_img_shape[::-1]
                                                                       , 5)

        """
        NOTES:
        - This could probably benefit from having generators built in. 
            Would allow iterating over images in loops. 
            You're welcome to add that in as a new function if you would like
        """

    def get_cam0(self, frame_num, rectify=False):
        r"""Returns the left image at the given frame"""
        assert frame_num <= self.num_frames, "Frame number greater than number of files"
        assert len(self.left_image_files) > 0, f"No image files found at {self.left_image_path}"

        im = os.path.join(self.left_image_path, self.left_image_files[frame_num])
        if not rectify:
            img = cv2.imread(im)
        else:
            img = cv2.remap(cv2.imread(im), self.left_mapx, self.left_mapy, cv2.INTER_LINEAR)
        return img

    def get_cam1(self, frame, rectify=False):
        r"""Returns the right image at the given frame"""
        assert frame <= self.num_frames, "Frame number greater than number of files"
        assert len(self.right_image_files) > 0, f"No image files found at {self.right_image_path}"

        im = os.path.join(self.right_image_path, self.right_image_files[frame])
        if not rectify:
            img = cv2.imread(im)
        else:
            img = cv2.remap(cv2.imread(im), self.right_mapx, self.right_mapy, cv2.INTER_LINEAR)
        return img

    def get_lidar(self, frame):
        r"""Returns the lidar points at the given frame"""
        assert frame <= self.num_frames, "Frame number greater than number of files"
        assert len(self.lidar_files) > 0, f"No Lidar files found at {self.lidar_path}"
        file = os.path.join(self.lidar_path, self.lidar_files[frame])
        return np.load(file)


def load_calib(left_calib, right_calib):
    with open(left_calib) as file:
        left_vars = yaml.load(file, Loader=yaml.FullLoader)
    with open(right_calib) as file:
        right_vars = yaml.load(file, Loader=yaml.FullLoader)

    cam = {}
    cam["cam0_img_shape"] = (left_vars["image_height"], left_vars["image_width"])
    cam["cam0_distortion_coefficients"] = np.array(left_vars['distortion_coefficients']['data']).reshape(
        (left_vars['distortion_coefficients']['rows'], left_vars['distortion_coefficients']['cols']))
    cam["cam0_rectification_matrix"] = np.array(left_vars['rectification_matrix']['data']).reshape(
        (left_vars['rectification_matrix']['rows'], left_vars['rectification_matrix']['cols']))
    cam["cam0_projection_matrix"] = np.array(left_vars['projection_matrix']['data']).reshape(
        (left_vars['projection_matrix']['rows'], left_vars['projection_matrix']['cols']))
    cam["cam0_camera_matrix"] = np.array(left_vars['camera_matrix']['data']).reshape(
        (left_vars['camera_matrix']['rows'], left_vars['camera_matrix']['cols']))

    cam["cam1_img_shape"] = (right_vars["image_height"], right_vars["image_width"])
    cam["cam1_distortion_coefficients"] = np.array(right_vars['distortion_coefficients']['data']).reshape(
        (right_vars['distortion_coefficients']['rows'], right_vars['distortion_coefficients']['cols']))
    cam["cam1_rectification_matrix"] = np.array(right_vars['rectification_matrix']['data']).reshape(
        (right_vars['rectification_matrix']['rows'], right_vars['rectification_matrix']['cols']))
    cam["cam1_projection_matrix"] = np.array(right_vars['projection_matrix']['data']).reshape(
        (right_vars['projection_matrix']['rows'], right_vars['projection_matrix']['cols']))
    cam["cam1_camera_matrix"] = np.array(right_vars['camera_matrix']['data']).reshape(
        (right_vars['camera_matrix']['rows'], right_vars['camera_matrix']['cols']))

    cam["baseline"] = np.array([(-cam["cam1_projection_matrix"][0, 3] / cam["cam1_camera_matrix"][0, 0]), 0, 0])

    calib = namedtuple("calib", cam.keys())(*cam.values())
    return calib


def timesync_dataset(path_to_data, simulate_removal=True, seconds_tolerance=0.1, timestamp_resolution=1e-9,
                     verbose=True):
    r"""
    This is to check that an extracted dataset is timesynced.
    It will remove those files that do not sync to within the required tolerance.
    """
    import time
    # list the folders that contain data in the husky DatasetHandler() format
    import glob
    dataset_paths = glob.glob(path_to_data + "*/image_00/data/")

    # for each folder:
    for d in dataset_paths:
        set_dir = d.rsplit('/image_00/data/')[0]

        # make a timesynced DatasetHandler() object
        husky_var = DatasetHandler(set_dir)

        # get the list of all the files
        l_imgs = os.listdir(husky_var.left_image_path)
        r_imgs = os.listdir(husky_var.right_image_path)
        lid_files = os.listdir(husky_var.lidar_path)

        # get the list of files that consitute the un-synced data
        l_diff = set(husky_var.left_image_files).symmetric_difference(set(os.listdir(husky_var.left_image_path)))
        r_diff = set(husky_var.right_image_files).symmetric_difference(set(os.listdir(husky_var.right_image_path)))
        lid_diff = set(husky_var.lidar_files).symmetric_difference(set(os.listdir(husky_var.lidar_path)))

        # if simulating:
        #       Print out the summary of files that do and do not sync
        #       make output file of the files to be removed
        #           -   Would be nice to have it as a shell script that could just be run to remove those files

        if simulate_removal:
            print(f"Simulating removal:\n"
                  f"Found:\n"
                  f"\t - {len(husky_var.left_image_files)} files that are synced.")
            if len(l_diff):
                print(f"\t - {len(l_diff)} unsynced left files of {len(l_imgs)}.")
            if len(r_diff):
                print(f"\t - {len(r_diff)} unsynced right files of {len(r_imgs)}.")
            if len(lid_diff):
                print(f"\t - {len(lid_diff)} unsynced lidar files of {len(lid_files)}.")
            print(f"Run without the simulate_removal flag to remove.\n"
                  f"-------------------------------------------------------\n")

            print("Preview of removal:")
            if len(l_diff):
                print(f"Removing left img   -> {os.path.join(husky_var.left_image_path, list(l_diff)[0])}")
            if len(r_diff):
                print(f"Removing right img  -> {os.path.join(husky_var.right_image_path, list(r_diff)[0])}")
            if len(lid_diff):
                print(f"Removing lidar scan -> {os.path.join(husky_var.lidar_path, list(lid_diff)[0])}")

        else:
            # If not simulating:
            #       remove those files that are not in the time synced list

            print("Preview of removal:")
            if len(l_diff):
                print(f"Removing left img   -> {os.path.join(husky_var.left_image_path, list(l_diff)[0])}")
            if len(r_diff):
                print(f"Removing right img  -> {os.path.join(husky_var.right_image_path, list(r_diff)[0])}")
            if len(lid_diff):
                print(f"Removing lidar scan -> {os.path.join(husky_var.lidar_path, list(lid_diff)[0])}")

            print("Press ctrl+c if this is wrong\n \t.... you have 5 seconds")
            time.sleep(5)

            if len(l_diff):
                print("Removing left files")
                print()
                for l in tqdm.tqdm(l_diff):
                    os.remove(os.path.join(husky_var.left_image_path, l))
                print()

            if len(r_diff):
                print("Removing right files")
                print()
                for r in tqdm.tqdm(r_diff):
                    os.remove(os.path.join(husky_var.right_image_path, r))
                print()

            if len(lid_diff):
                print("Removing lidar files")
                print()
                for li in tqdm.tqdm(lid_diff):
                    os.remove(os.path.join(husky_var.lidar_path, li))
                print()

            # print summary of what happened
            print("Summary of removal:")
            print(f"\t - Removed {len(l_imgs) - len(os.listdir(husky_var.left_image_path))} left images")
            print(f"\t - Removed {len(r_imgs) - len(os.listdir(husky_var.right_image_path))} left images")
            print(f"\t - Removed {len(lid_files) - len(os.listdir(husky_var.lidar_path))} left images")

    # repeat for the rest of the folders with data.

    return 0


def bagfile_to_dataset(path_bagfile, path_output_dataset, left_calib_path = CAMERA_LEFT_CALIB_FILE,
                       right_calib_path = CAMERA_RIGHT_CALIB_FILE, custom_date=False, use_csv = False,
                       images=True, rectify_and_undistort = True,
                       lidar=True, verbose=True, overwrite=False):
    # Topics:
    #     / velodyne_points
    #     "/camera/image_left/image_raw"
    #     "/camera/image_right/image_raw"

    if images:
        left_topic = "/camera/image_left/image_raw"
        right_topic = "/camera/image_right/image_raw"
    if lidar:
        lidar_topic = "/velodyne_points/"

    if rectify_and_undistort:
        if not os.path.exists(left_calib_path):
            print("The calibration yaml file you entered for the left camera does not exist.")
        if not os.path.exists(right_calib_path):
            print("The calibration yaml file you entered for the left camera does not exist.")

        calib = load_calib(left_calib_path, right_calib_path)
        # For distortion:
        left_mapx, left_mapy = cv2.initUndistortRectifyMap(calib.cam0_camera_matrix,
                                                                     calib.cam0_distortion_coefficients.squeeze(),
                                                                     None,
                                                                     calib.cam0_projection_matrix,
                                                                     calib.cam0_img_shape[::-1]
                                                                     , 5)

        right_mapx, right_mapy = cv2.initUndistortRectifyMap(calib.cam1_camera_matrix,
                                                                       calib.cam1_distortion_coefficients.squeeze(),
                                                                       None,
                                                                       calib.cam1_projection_matrix,
                                                                       calib.cam1_img_shape[::-1]
                                                                       , 5)

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
            split = os.path.splitext(bag_name)[0].split('-')
            # date = dparser.parse(os.path.splitext(bag_name)[0], fuzzy=True).strftime("%Y_%m_%d_%H_%M")
            date = dparser.parse(os.path.splitext(bag_name)[0], fuzzy=True).strftime("%Y_%m_%d") + \
                   f"_{split[-3]}_{split[-2]}_{split[-1]}"

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
            with open(os.path.join(timestamp_path, "timestamps.txt"), 'w') as left_timestamps:
                for _ in tqdm.trange(len_left):
                    # for topic, msg, t in ros_bag.read_messages(topics=left_topic):
                    topic, msg, t = next(left_gen)
                    left_timestamps.write(str(t) + '\n')
                    bag_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                    if rectify_and_undistort:
                        out_img = cv2.remap(bag_img, left_mapx, left_mapy, cv2.INTER_LINEAR)
                    else:
                        out_img = bag_img
                    cv2.imwrite(os.path.join(left_im_path, f"{str(t)}.png"), out_img)

            if verbose: print("\t\t- Extracting Right Images:")
            right_gen = ros_bag.read_messages(topics=right_topic)


            # Timestamp writer
            timestamp_path, _ = os.path.split(right_im_path)
            with open(os.path.join(timestamp_path, "timestamps.txt"), 'w') as right_timestamps:
                for _ in tqdm.trange(len_right):
                    # for topic, msg, t in ros_bag.read_messages(topics=topic_right):
                    topic, msg, t = next(right_gen)

                    right_timestamps.write(str(t) + '\n')

                    bag_img = np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.width,-1)

                    if rectify_and_undistort:
                        out_img = cv2.remap(bag_img, right_mapx, right_mapy, cv2.INTER_LINEAR)
                    else:
                         out_img = bag_img

                    cv2.imwrite(os.path.join(right_im_path, f"{str(t)}.png"), out_img)
            if verbose: print("\n\t......Done extracting images\n")

        if lidar:
            if verbose: print("\n\t- Extracting Lidar Data")
            len_lidar = ros_bag.get_message_count(lidar_topic)
            lidar_gen = ros_bag.read_messages(topics=lidar_topic)

            for _ in tqdm.trange(len_lidar):
                topic, msg, t = next(lidar_gen)
                pts = rnp.point_cloud2.pointcloud2_to_xyz_array(msg)

                if use_csv:
                    lidar_file = os.path.join(velodyne_path, f"{t}.csv")
                    np.savetxt(lidar_file, pts, delimiter=',')
                else:
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

    return 0


if __name__ == "__main__":
    print("Running tests")

    print("Test extraction")
    # bagfile_path = "/media/kats/Katsoulis3/Datasets/Husky/gucci_bags/Calibration and Lab"
    # output_path = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/Calibration and Lab"

    bagfile_path = '/media/kats/Katsoulis3/Datasets/Husky/gucci_bags/UCT Outdoor/old_zoo'
    output_path = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo_new/"

    bagfile_to_dataset(bagfile_path, output_path, lidar=True, overwrite=False)

    # timesync_dataset(output_path, simulate_removal=False)
