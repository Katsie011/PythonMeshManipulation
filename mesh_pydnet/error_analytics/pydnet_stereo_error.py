"""
Getting the error on pydnet predictions compared to stereo
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import utilities.HuskyDataHandler as husky
import os
import sys
import tensorflow as tf
import argparse
import tqdm
import time



sys.path.append("/home/kats/Code/aru_sil_py/reconstruction/mesh_pydnet/pydnet")
from utils import *
from pydnet import *

import utilities.ImgFeatureExtactorModule as img_ft
from reconstruction.mesh_pydnet.pydnet_VO import get_dataset, init_pydepth


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--dataset', type=str, help='dataset to train on, kitti, or Husky', default='Husky')
parser.add_argument('--datapath', type=str, help='path to the data',
                    default=r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/RouteC/2022_05_03_14_09_01")  # required=True)
# parser.add_argument('--output_directory', type=str,
#                     help='output directory for test disparities, if empty outputs to checkpoint folder',
#                     default=output_directory)
parser.add_argument('--checkpoint_dir', type=str, help='path to a specific checkpoint to load',
                    default='/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/mesh_pydnet/checkpoints/Husky10K/Husky')
parser.add_argument('--resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--save_predictions', type=bool, help="save predicted disparities to output directory",
                    default=True)
parser.add_argument('--width', dest='width', type=int, default=1280, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=768, help='height of input images')

args = parser.parse_args()


sift = img_ft.FeatureDetector(det_type='sift', max_num_ft=2000)
orb = img_ft.FeatureDetector(det_type='orb', max_num_ft=2000)

MAX_DISTANCE = 100


def get_stereo_disp(imgl, imgr, det=sift):
    kpsl, desl = det.detect(imgl)
    kpsr, desr = det.detect(imgr)
    good_matches = det.get_matches(desl, desr, lowes_ratio=0.9)

    train_idxs = np.zeros(len(good_matches), dtype=int)
    query_idxs = np.zeros(len(good_matches), dtype=int)

    for i, m in enumerate(good_matches):
        train_idxs[i] = m.trainIdx
        query_idxs[i] = m.queryIdx

    kpsl = np.array(kpsl, dtype=object)
    kpsr = np.array(kpsr, dtype=object)

    pts_l = det.kp_to_pts(kpsl[query_idxs])
    pts_r = det.kp_to_pts(kpsr[train_idxs])
    disp = (pts_l[:, 0] - pts_r[:, 0])
    return disp, pts_l, pts_r

def pydnet_vs_stereo_depth(prediction:np.ndarray, frame: int, dataset: husky.DatasetHandler, det=sift, use_depth = False, return_pts = False, epipolar_window=2):
    imgl= dataset.get_cam0(frame)
    imgr = dataset.get_cam1(frame)
    disp, pts_l, pts_r = get_stereo_disp(imgl, imgr, det=det)
    good_disp_idx = (disp > 0) & (np.abs(pts_l[:,1]-pts_r[:,1]) < epipolar_window)
    good_pts_l = pts_l[good_disp_idx]
    good_pts_r = pts_r[good_disp_idx]
    good_disp = disp[good_disp_idx]

    pred_pts = prediction[good_pts_l[:, 1].astype(int), good_pts_l[:, 0].astype(int)]

    if use_depth:
        stereo_depth = dataset.calib.baseline[0] * (
                    dataset.calib.cam0_camera_matrix[0, 0] * dataset.img_shape[1]) / good_disp
        e = pred_pts - stereo_depth
    else:
        e=pred_pts-good_disp

    se = e**2
    mse = se.mean()
    print(f"\nGot {len(good_disp)} good points from SIFT\n"
          f"\tSSE stereo-to-prediction: {se.sum():.5f}, \sigma ={se.std()}\n"
          f"\tMSE stereo-to-prediction: {mse:.5f}, \sigma ={mse.std()}\n"
          f"\tAvg E stereo-to-prediction: {e.mean():.5f}, \sigma ={e.std()}\n"
          f"\tAvg stereo disparity: {good_disp.mean():.5f}, \sigma ={good_disp.std()}\n"
          f"\tAvg predicted disparity: {pred_pts.mean():.5f}, \sigma ={pred_pts.std()}\n"
          )
    if return_pts:
        return mse, good_pts_l
    return mse




#
def predict_dataset_get_stereo_e(dataset, plots=True, verbose=False, training_size=(256, 512), max_frames=None, get_orb=True,
                    get_sift=True):
    # if not get_orb and not get_sift:
    #     print("No feature detector chosen. Returning None")
    #     return None
    if max_frames is None:
        max_frames = dataset.num_frames - 1
    if get_sift:
        x_sift, y_sift = np.zeros(max_frames), np.zeros(max_frames)
        mse_sift = np.zeros(max_frames)
    if get_orb:
        x_orb, y_orb = np.zeros(max_frames), np.zeros(max_frames)
        mse_orb = np.zeros(max_frames)


    # input_shape = dataset.img_shape
    train_height, train_width = training_size
    with tf.Graph().as_default():
        placeholders, model, init, loader, saver = init_pydepth()
        show_flag = True

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            for counter in tqdm.trange(max_frames):
                # for counter in tqdm.trange(dataset.num_frames):
                start_loop_time = time.time()
                imgl = cv2.cvtColor(dataset.get_cam0(counter), cv2.COLOR_BGR2RGB)

                img = cv2.resize(imgl, (train_width, train_height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)

                # Getting training sized image depth
                start_pred = time.time()
                disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end_pred = time.time()
                time_pred = start_pred - end_pred
                disparity = disp[0, :, :, 0].squeeze() * 0.3 * train_width
                # disparity[disparity<MAX_DISPARITY] = MAX_DISPARITY
                depth = dataset.calib.baseline[0] * (dataset.calib.cam0_camera_matrix[0, 0] * train_width / dataset.img_shape[1]) / disparity
                depth[depth > MAX_DISTANCE] = MAX_DISTANCE
                fullsize_depth = cv2.resize(depth, (dataset.img_shape[1], dataset.img_shape[0]))
                fullsize_disparity = cv2.resize(disparity,(dataset.img_shape[1], dataset.img_shape[0])) * dataset.img_shape[1]/train_width
                if verbose: print(f"Time to predict image of size {training_size}: {time_pred}")

                if get_sift:
                    mse_sift[counter], sift_pts = pydnet_vs_stereo_depth(prediction=fullsize_disparity, frame=counter,
                                                                         dataset=dataset, det=sift, return_pts=True,
                                                                         use_depth=False)
                    if plots:
                        sift_im = cv2.cvtColor(imgl, cv2.COLOR_RGB2BGR)
                        for c in sift_pts.astype(int):
                            sift_im = cv2.circle(sift_im, c, 5, (0, 0, 255), 2)

                        cv2.imshow("Sift Points", cv2.resize(sift_im, (dataset.img_shape[1]//2, dataset.img_shape[0]//2)))

                if get_orb:
                    mse_orb[counter], orb_pts = pydnet_vs_stereo_depth(prediction=fullsize_disparity, frame=counter,
                                                                       dataset=dataset, det=orb, return_pts=True,
                                                                       use_depth=False)
                    if plots:
                        orb_im = cv2.cvtColor(imgl, cv2.COLOR_RGB2BGR)
                        for c in orb_pts.astype(int):
                            orb_im = cv2.circle(orb_im, c, 5, (0, 0, 255), 2)

                        cv2.imshow("Orb Points", cv2.resize(orb_im, (dataset.img_shape[1]//2, dataset.img_shape[0]//2)))



                end_loop_time = time.time()
                print(
                    f"\nTook {end_loop_time - start_loop_time:.3f}s to run one iteration with SIFT and ORB VO together")

                if plots:
                    img_and_depth = np.hstack((cv2.cvtColor(imgl, cv2.COLOR_RGB2BGR), cv2.applyColorMap(
                        (fullsize_disparity*255/fullsize_disparity.max()).astype(np.uint8),cv2.COLORMAP_PLASMA)))
                    cv2.imshow("Input and disparity", cv2.resize(img_and_depth, (dataset.img_shape[1], dataset.img_shape[0]//2)))
                    cv2.waitKey(20)

    # Returning points
    if get_sift and get_orb:
        return mse_orb, mse_sift
    elif get_sift and not get_orb:
        return mse_sift
    elif get_orb:
        return mse_orb

if __name__ == "__main__":
    # data_dir = "/home/kats/Datasets/Whitelab/Dataset_Structures/2022_07_04_10_26_41"
    data_dir = "/home/kats/Datasets/Route A/2022_07_06_10_48_24"
    dataset_obj = husky.DatasetHandler(data_dir, time_tolerance=0.5)
    get_stereo_disp(dataset_obj.get_cam0(0), dataset_obj.get_cam1(0))
    max_frames = 500

    es = predict_dataset_get_stereo_e(dataset=dataset_obj, get_sift=True, get_orb=False, plots=True, max_frames=max_frames)
    plt.plot(es)
    plt.title(f"MSE between stereo and predicted depth by frame. Avg mse: {es.mean():.4f}")
    plt.xlabel("Frame number")
    plt.ylabel("MSE")
    plt.show()
    print("done")
