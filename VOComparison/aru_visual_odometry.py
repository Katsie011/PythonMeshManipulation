"""
This script compares the accuracy of visual odometry using stereo images vs using monocular depth predictions.
This gives an understanding of whether the monocular network is accurate enough for geometric reconstruction tasks.

Authur: Michael Katsoulis

"""
import aru_core_lib.aru_py_vo as aru_py_vo
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm
import ModularFiles.HuskyDataHandler as husky
import ModularFiles.ImgFeatureExtactorModule as ft

sift = ft.FeatureDetector(det_type='sift', max_num_ft=500)
pyvo = aru_py_vo.PyVO(
    '/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/VOComparison/config/vo_config.yaml')


def read_predictions(prediction_path):
    pass


def get_vo(frame0, frame1, data, pyvo=pyvo):
    s0l = cv2.cvtColor(data.get_cam0(frame0, rectify=True), cv2.COLOR_RGB2GRAY)
    s0r = cv2.cvtColor(data.get_cam1(frame0, rectify=True), cv2.COLOR_RGB2GRAY)
    s1r = cv2.cvtColor(data.get_cam1(frame1, rectify=True), cv2.COLOR_RGB2GRAY)
    s1l = cv2.cvtColor(data.get_cam0(frame1, rectify=True), cv2.COLOR_RGB2GRAY)
    t1 = data.left_times[frame1].astype(np.int64)
    t0 = data.left_times[frame0].astype(np.int64)

    return pyvo.stereo_odometry(s0l, s0r, t0, s1l, s1r, t1)


def generate_transform_path(transforms: np.array):
    coords = np.eye(4)
    x_vals = []
    y_vals = []
    for transform in transforms:
        coords = coords @ transform
        # z-axis is fwd for img coords, x-axis is fwd in body coord sys
        x_vals.append(coords[2][3])
        # x-axis is right img coords, y-axis is left in body coord sys
        y_vals.append(-coords[0][3])

    return [x_vals, y_vals]


def get_depth_predictions(predictions_directory, dataset):
    out_files = os.listdir(predictions_directory)

    unsynced_pred_files = np.asarray(os.listdir(predictions_directory), dtype=object)
    unsynced_times = -1 * np.ones(len(unsynced_pred_files))
    pred_files = np.zeros(len(dataset.left_image_files), dtype=object)
    pred_times = -1 * np.ones(dataset.num_frames)
    for i, f in enumerate(unsynced_pred_files):
        # Getting times
        unsynced_times[i] = int(os.path.splitext(f)[0])

    for i, lt in enumerate(dataset.left_times):
        # Need to get the first image where t_right > t_left
        # Right images are published after left but aquired simultaneously

        # getting the time difference w.r.t. current right time
        difference = unsynced_times - lt
        pred_ind = np.argmin(np.abs(difference))  # minimum along left times

        if difference[pred_ind] * dataset.timestamp_resolution < dataset.time_tolerance:
            # getting the first frame where right is after left (first positive time)
            pred_files[i] = unsynced_pred_files[pred_ind]
            pred_times[i] = unsynced_times[pred_ind]

            # need to reassure that this matches all files and that right images are not published after left.

            # removing the matched file from the right files that still need to be matched
            # this is to avoid duplicate matches
            ind = np.ones(len(unsynced_pred_files), dtype=bool)
            ind[pred_ind] = False

            unsynced_pred_files = unsynced_pred_files[ind]
            unsynced_times = unsynced_times[ind]

        else:
            print(f"Could not match {[dataset.left_image_files][i]} with a depth prediction file")

    return pred_files, pred_times


def depth_pred_vo(depth, img0, img1, K, det=sift, vo=pyvo):
    kps0, des0 = det.detect(img0)
    kps1, des1 = det.detect(img1)

    matches = det.get_matches(des0, des1)

    # Getting the indicies of the matches
    train_idx = []
    query_idx = []
    for m in matches:
        train_idx.append(m.trainIdx)
        query_idx.append(m.queryIdx)

    # Getting coordinates of the keypoints at the matches
    u0, v0 = det.kp_to_pts(kps0)[train_idx].T
    uv1 = det.kp_to_pts(kps1)[query_idx]

    # Extracting camera parameters
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # getting 3d points in frame 0
    pt_u0 = np.floor(u0*depth.shape[1]/img0.shape[1]).astype(int)
    pt_v0 = np.floor(v0*depth.shape[0]/img0.shape[0]).astype(int)
    x_cam = (u0 - cx) * depth[pt_v0, pt_u0] / fx
    y_cam = (v0 - cy) * depth[pt_v0, pt_u0] / fy
    z_cam = depth[pt_v0, pt_u0]

    world_x = z_cam
    world_y = -x_cam
    world_z = -y_cam

    pts3d = np.stack((world_x, world_y, world_z), axis=1)


    # get VO estimation from 3d points in frame 0 and 2d pts in frame 1
    return vo.motion_estimation(pts3d.T, uv1.T)


if __name__ == "__main__":
    data_dir = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/RouteC/2022_05_03_14_09_01"
    output_directory = os.path.join(data_dir, 'predictions/output/image_00/data')
    input_directory = os.path.join(data_dir, 'predictions/input/image_00/data')
    dataset = husky.DatasetHandler(data_dir)

    # Trs = np.zeros((dataset.num_frames - 1, 4, 4))
    #
    # for i in tqdm.trange(dataset.num_frames - 1):
    #     Trs[i] = get_vo(i, i + 1, dataset)
    #
    # x_stereo, y_stereo = generate_transform_path(Trs)
    # plt.plot(x_stereo, y_stereo)
    # plt.title("Stereo Visual Odometry results")
    # plt.grid()
    # plt.show()
    #
    # print("Done")

    depth_filenames, depth_times = get_depth_predictions(output_directory, dataset)

    for i in range(dataset.num_frames):
        d = depth_filenames[i]
        depth = cv2.imread(os.path.join(output_directory, d), cv2.IMREAD_GRAYSCALE)
        im0 = dataset.get_cam0(i)
        im1 = dataset.get_cam1(i)
        K = dataset.calib.cam0_camera_matrix

        print(depth_pred_vo(depth, im0, im1, K))
