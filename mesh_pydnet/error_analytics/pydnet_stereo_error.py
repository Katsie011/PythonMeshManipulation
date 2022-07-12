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
parser.add_argument('--vo_config', type=str, help="Path to config for VO",
                    default="/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/VOComparison/config/vo_config.yaml")
parser.add_argument('--width', dest='width', type=int, default=1280, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=768, help='height of input images')

args = parser.parse_args()

# pyvo = aru_py_vo.PyVO(args.vo_config)

sift = img_ft.FeatureDetector(det_type='sift', max_num_ft=2000)
orb = img_ft.FeatureDetector(det_type='orb', max_num_ft=2000)
from reconstruction.VOComparison.aru_visual_odometry import pyvo
import reconstruction.VOComparison.aru_visual_odometry as aru_vo_module

MAX_DISTANCE = 100


def get_stereo_disp(imgl, imgr, det=sift):
    kpsl, desl = det.detect(imgl)
    kpsr, desr = det.detect(imgr)
    good_matches = det.get_matches(desl, desr, lowes_ratio=0.9)

    idx_l, idx_r = det.matches_and_keypts_to_good_idxs(good_matches=good_matches, kps_l=kpsl, kps_r=kpsr)
    pts_l = det.kp_to_pts(np.array(kpsl, dtype=object)[idx_l])
    pts_r = det.kp_to_pts(np.array(kpsr, dtype=object)[idx_r])

    # train_idxs = np.zeros(len(good_matches), dtype=int)
    # query_idxs = np.zeros(len(good_matches), dtype=int)
    #
    # for i, m in enumerate(good_matches):
    #     train_idxs[i] = m.trainIdx
    #     query_idxs[i] = m.queryIdx
    #
    # kpsl = np.array(kpsl, dtype=object)
    # kpsr = np.array(kpsr, dtype=object)
    #
    # pts_l = det.kp_to_pts(kpsl[query_idxs])
    # pts_r = det.kp_to_pts(kpsr[train_idxs])
    disp = (pts_l[:, 0] - pts_r[:, 0])
    return disp, pts_l, pts_r


def get_aru_core_stereo_pts(imgl, imgr, vo_obj=pyvo):
    if len(imgl.shape) == 3:
        imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
        imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    return vo_obj.stereo_matches(imgl, imgr)


def pydnet_vs_aru_core_pts(prediction: np.ndarray, frame: int, dataset: husky.DatasetHandler, use_depth=False,
                           return_mse=True, return_pts=False, epipolar_window=2, verbose=False):
    imgl = dataset.get_cam0(frame)
    imgr = dataset.get_cam1(frame)
    imgl_p1 = dataset.get_cam0(frame+1)
    imgr_p1 = dataset.get_cam1(frame+1)

    if imgl.shape[:2] != prediction.shape:
        # imgl = cv2.resize(imgl, prediction.shape[::-1])
        # imgr = cv2.resize(imgr, prediction.shape[::-1])
        # imgl_p1 = cv2.resize(imgl_p1, prediction.shape[::-1])
        # imgr_p1 = cv2.resize(imgr_p1, prediction.shape[::-1])
        prediction = cv2.resize(prediction, imgl.shape[:2][::-1])

    Tr_stereo, pts_l, pts_r, pts_lp1 = pyvo.stereo_odometry(cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY),
                                                     cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY), dataset.left_times[frame].astype(int),
                                                     cv2.cvtColor(imgl_p1, cv2.COLOR_BGR2GRAY),
                                                     cv2.cvtColor(imgr_p1, cv2.COLOR_BGR2GRAY),
                                                     dataset.left_times[frame+1].astype(int))


    # pts_l, pts_r = get_aru_core_stereo_pts(imgl, imgr)
    disp = pts_l[:, 0] - pts_r[:, 0]
    # good_disp_idx = (np.abs(pts_l[:, 1] - pts_r[:, 1]) < epipolar_window)
    # good_pts_l = pts_l[good_disp_idx]
    # good_pts_r = pts_r[good_disp_idx]
    # good_disp = disp[good_disp_idx]
    good_pts_l = pts_l
    good_pts_r = pts_r
    good_disp = disp
    pred_pts = prediction[good_pts_l[:, 1].astype(int), good_pts_l[:, 0].astype(int)]

    Tr_pydnet = aru_vo_module.motion_est_from_ft_pts(pts_l, pts_lp1, depth = pred_pts, K=dataset.calib.cam0_camera_matrix, vo=pyvo,
                                       verify_trans=True, max_step_d=2)

    if use_depth:
        stereo_depth = dataset.calib.baseline[0] * (
            dataset.calib.cam0_camera_matrix[0, 0]) / good_disp
        e = pred_pts - stereo_depth
    else:
        e = pred_pts - good_disp

    se = e ** 2
    mse = se.mean()

    if verbose:
        print(f"\nGot {len(good_disp)} good points from ARU-CORE\n"
          f"\tSSE stereo-to-prediction: {se.sum():.5f}, \sigma ={se.std()}\n"
          f"\tMSE stereo-to-prediction: {mse:.5f}, \sigma ={mse.std()}\n"
          f"\tAvg E stereo-to-prediction: {e.mean():.5f}, \sigma ={e.std()}\n"
          f"\tAvg stereo disparity: {good_disp.mean():.5f}, \sigma ={good_disp.std()}\n"
          f"\tAvg predicted disparity: {pred_pts.mean():.5f}, \sigma ={pred_pts.std()}\n"
          )

    if return_mse:
        e_ret = mse
    else:
        e_ret = np.abs(e)

    if return_pts:
        return e_ret, np.mean(good_disp/pred_pts), np.hstack((good_pts_l, good_disp.reshape((-1, 1)))), Tr_pydnet, Tr_stereo
    return e_ret, Tr_pydnet, Tr_stereo


def pydnet_vs_stereo_depth(prediction: np.ndarray, frame: int, dataset: husky.DatasetHandler, det=sift, use_depth=False,
                           return_mse=True, return_pts=False, epipolar_window=2):
    imgl = dataset.get_cam0(frame)
    imgr = dataset.get_cam1(frame)

    if imgl.shape[:2] != prediction.shape:
        imgl = cv2.resize(imgl, prediction.shape[::-1])
        imgr = cv2.resize(imgr, prediction.shape[::-1])

    disp, pts_l, pts_r = get_stereo_disp(imgl, imgr, det=det)
    good_disp_idx = (disp > 0) & (np.abs(pts_l[:, 1] - pts_r[:, 1]) < epipolar_window)
    good_pts_l = pts_l[good_disp_idx]
    good_pts_r = pts_r[good_disp_idx]
    good_disp = disp[good_disp_idx]

    pred_pts = prediction[good_pts_l[:, 1].astype(int), good_pts_l[:, 0].astype(int)]

    if use_depth:
        stereo_depth = dataset.calib.baseline[0] * (
            dataset.calib.cam0_camera_matrix[0, 0]) / good_disp
        e = pred_pts - stereo_depth
    else:
        e = pred_pts - good_disp

    se = e ** 2
    mse = se.mean()
    print(f"\nGot {len(good_disp)} good points from SIFT\n"
          f"\tSSE stereo-to-prediction: {se.sum():.5f}, \sigma ={se.std()}\n"
          f"\tMSE stereo-to-prediction: {mse:.5f}, \sigma ={mse.std()}\n"
          f"\tAvg E stereo-to-prediction: {e.mean():.5f}, \sigma ={e.std()}\n"
          f"\tAvg stereo disparity: {good_disp.mean():.5f}, \sigma ={good_disp.std()}\n"
          f"\tAvg predicted disparity: {pred_pts.mean():.5f}, \sigma ={pred_pts.std()}\n"
          )

    if return_mse:
        e_ret = mse
    else:
        e_ret = np.abs(e)

    if return_pts:
        return e_ret, np.mean(good_disp/pred_pts),  np.hstack((good_pts_l, good_disp.reshape((-1, 1))))
    return e_ret


#
def predict_dataset_get_stereo_e(dataset, scaling_factor = 1, plots=True, verbose=False, training_size=(256, 512), max_frames=None,
                                 get_orb=False, get_sift=True, aru_core_pts = True):
    # if not get_orb and not get_sift:
    #     print("No feature detector chosen. Returning None")
    #     return None
    disp_max = 100
    if max_frames is None:
        max_frames = dataset.num_frames - 1
    if get_sift:
        coords_sift = np.eye(4)
        x_sift, y_sift = np.zeros(max_frames), np.zeros(max_frames)
        mse_sift = np.zeros(max_frames)
    if get_orb:
        coords_orb = np.eye(4)
        x_orb, y_orb = np.zeros(max_frames), np.zeros(max_frames)
        mse_orb = np.zeros(max_frames)
    if aru_core_pts:
        coords_core = np.eye(4)
        x_core, y_core = np.zeros(max_frames), np.zeros(max_frames)
        # mse_core = np.zeros(max_frames)
        core_es = []

    coords_pydnet = np.eye(4)
    x_pydnet, y_pydnet = np.zeros(max_frames), np.zeros(max_frames)

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
                # disparity = disp[0, :, :, 0].squeeze() * 0.3 * train_width
                disparity = disp[0, :, :, 0].squeeze() * 0.3 * train_width * scaling_factor

                depth = dataset.calib.baseline[0] * (
                        dataset.calib.cam0_camera_matrix[0, 0] * train_width / dataset.img_shape[1]) / disparity
                depth[depth > MAX_DISTANCE] = MAX_DISTANCE
                fullsize_depth = cv2.resize(depth, (dataset.img_shape[1], dataset.img_shape[0]))
                fullsize_disparity = cv2.resize(disparity, (dataset.img_shape[1], dataset.img_shape[0])) * \
                                     dataset.img_shape[1] / train_width
                if verbose: print(f"Time to predict image of size {training_size}: {time_pred}")
                #
                # if get_sift:
                #     # mse_sift[counter], sift_pts = pydnet_vs_stereo_depth(prediction=fullsize_disparity, frame=counter,
                #     #                                                      dataset=dataset, det=sift, return_pts=True,
                #     #                                                      use_depth=False)
                #     print("\nEval on training size:")
                #     frame_sift_e, sift_pts = pydnet_vs_stereo_depth(prediction=disparity, frame=counter,
                #                                                     dataset=dataset, det=sift, return_pts=True,
                #                                                     return_mse=False,
                #                                                     use_depth=False)
                #
                #     mse_sift[counter] = np.mean(frame_sift_e ** 2)
                #     if plots:
                #         disp_max = 50
                #         sift_disp_samples = disparity[sift_pts[:, 1].astype(int), sift_pts[:, 0].astype(int)]
                #         im_e_sift = cv2.cvtColor(cv2.resize(imgl, disparity.shape[::-1]), cv2.COLOR_RGB2BGR)
                #         im_disp_sift = cv2.cvtColor(cv2.resize(imgl, disparity.shape[::-1]), cv2.COLOR_RGB2BGR)
                #         im_disp_pred = cv2.cvtColor(cv2.resize(imgl, disparity.shape[::-1]), cv2.COLOR_RGB2BGR)
                #         scaled_e = (np.abs(frame_sift_e) * 255 / disp_max)
                #         scaled_e[scaled_e > 255] = 255
                #         e_colors = cv2.applyColorMap(scaled_e.astype(np.uint8), cv2.COLORMAP_JET)
                #         scaled_s_dis = (np.abs(sift_pts[:, 2]) * 255 / disp_max)
                #         scaled_s_dis[scaled_s_dis > 255] = 255
                #         stereo_disp_colors = cv2.applyColorMap(scaled_s_dis.astype(np.uint8), cv2.COLORMAP_INFERNO)
                #         scaled_disp_s = (np.abs(sift_disp_samples) * 255 / disp_max)
                #         scaled_disp_s[scaled_disp_s > 255] = 255
                #         pred_disp_colors = cv2.applyColorMap(scaled_disp_s.astype(np.uint8), cv2.COLORMAP_INFERNO)
                #         for i, c in enumerate(sift_pts.astype(int)):
                #             im_e_sift = cv2.circle(im_e_sift, c[:2], 2, e_colors[i].squeeze().tolist(), 2)
                #             im_disp_sift = cv2.circle(im_disp_sift, c[:2], 2, stereo_disp_colors[i].squeeze().tolist(),
                #                                       2)
                #             im_disp_pred = cv2.circle(im_disp_pred, c[:2], 2, pred_disp_colors[i].squeeze().tolist(), 2)
                #         cv2.imshow("Error sift to pred disparities (Jet --> red is bigger e)", im_e_sift)
                #         cv2.imshow("Sift disparities", im_disp_sift)
                #         cv2.imshow("Predicted disparities", im_disp_pred)


                if aru_core_pts:
                    frame_core_e, frame_core_ratio ,core_pts, Tr_pydnet, Tr_stereo = pydnet_vs_aru_core_pts(prediction=fullsize_depth, frame=counter,
                                                                    dataset=dataset, return_pts=True,
                                                                    return_mse=False,
                                                                    use_depth=False, verbose=False)

                    # frame_core_e, core_pts = pydnet_vs_aru_core_pts(prediction=disparity, frame=counter,
                    #                                                 dataset=dataset, return_pts=True,
                    #                                                 return_mse=False,
                    #                                                 use_depth=False)

                    coords_core = coords_core @ Tr_stereo
                    # z-axis is fwd for img coords, x-axis is fwd in body coord sys
                    x_core[counter] = coords_core[2][3]
                    # x-axis is right img coords, y-axis is left in body coord sys
                    y_core[counter] = -coords_core[0][3]

                    coords_pydnet = coords_pydnet @ Tr_pydnet
                    # z-axis is fwd for img coords, x-axis is fwd in body coord sys
                    x_pydnet[counter] = coords_pydnet[2][3]
                    # x-axis is right img coords, y-axis is left in body coord sys
                    y_pydnet[counter] = -coords_pydnet[0][3]

                    # mse_core[counter] = np.mean(frame_core_e ** 2)
                    core_es.append([np.mean(frame_core_e ** 2), frame_core_e.mean(), frame_core_ratio])
                    print(f"\nFor frame{counter}:\n"
                          f"\tAvg E = {frame_core_e.mean():.3f}\n"
                          f"\tMSE   = {np.mean(frame_core_e ** 2):.3f}\n"
                          f"\tScale = {frame_core_ratio}")


                    if plots:
                        core_pts[:,:2] = core_pts[:,:2]*disparity.shape[1]/fullsize_disparity.shape[1]
                        core_disp_samples = disparity[core_pts[:, 1].astype(int), core_pts[:, 0].astype(int)]
                        im_e_core = cv2.cvtColor(cv2.resize(imgl, disparity.shape[::-1]), cv2.COLOR_RGB2BGR)
                        im_disp_core = cv2.cvtColor(cv2.resize(imgl, disparity.shape[::-1]), cv2.COLOR_RGB2BGR)
                        im_disp_pred = cv2.cvtColor(cv2.resize(imgl, disparity.shape[::-1]), cv2.COLOR_RGB2BGR)
                        scaled_e = (np.abs(frame_core_e) * 255 / disp_max)
                        scaled_e[scaled_e > 255] = 255
                        e_colors = cv2.applyColorMap(scaled_e.astype(np.uint8), cv2.COLORMAP_JET)
                        scaled_s_dis = (np.abs(core_pts[:, 2]) * 255 / disp_max)
                        scaled_s_dis[scaled_s_dis > 255] = 255
                        stereo_disp_colors = cv2.applyColorMap(scaled_s_dis.astype(np.uint8), cv2.COLORMAP_INFERNO)
                        scaled_disp_s = (np.abs(core_disp_samples) * 255 / disp_max)
                        scaled_disp_s[scaled_disp_s > 255] = 255
                        pred_disp_colors = cv2.applyColorMap(scaled_disp_s.astype(np.uint8), cv2.COLORMAP_INFERNO)
                        for i, c in enumerate(core_pts.astype(int)):
                            im_e_core = cv2.circle(im_e_core, c[:2], 2, e_colors[i].squeeze().tolist(), 2)
                            im_disp_core = cv2.circle(im_disp_core, c[:2], 2, stereo_disp_colors[i].squeeze().tolist(),
                                                      2)
                            im_disp_pred = cv2.circle(im_disp_pred, c[:2], 2, pred_disp_colors[i].squeeze().tolist(), 2)
                        cv2.imshow("Error core to pred disparities (Jet --> red is bigger e)", im_e_core)
                        cv2.imshow("Sift disparities", im_disp_core)
                        cv2.imshow("Predicted disparities", im_disp_pred)

                # if get_orb:
                #     print("\nEval on training size:")
                #
                #     mse_orb[counter], orb_pts = pydnet_vs_stereo_depth(prediction=disparity, frame=counter,
                #                                                        dataset=dataset, det=orb, return_pts=True,
                #                                                        use_depth=False)
                #
                #     # mse_orb[counter], orb_pts = pydnet_vs_stereo_depth(prediction=fullsize_disparity, frame=counter,
                #     #                                                    dataset=dataset, det=orb, return_pts=True,
                #     #                                                    use_depth=False)
                #     if plots:
                #         orb_im = cv2.cvtColor(cv2.resize(imgl, disparity.shape[::-1]), cv2.COLOR_RGB2BGR)
                #         for c in orb_pts.astype(int):
                #             orb_im = cv2.circle(orb_im, c, 2, (0, 0, 255), 2)
                #
                #         cv2.imshow("Orb Points", orb_im)

                end_loop_time = time.time()
                print(
                    f"\nTook {end_loop_time - start_loop_time:.3f}s to run one iteration with SIFT and ORB VO together")

                if plots:
                    if get_sift:
                        img_and_depth = np.hstack((cv2.cvtColor(cv2.resize(im_disp_sift, disparity.shape[::-1]),
                                                                cv2.COLOR_RGB2BGR), cv2.applyColorMap(
                            (disparity * 255 / disparity.max()).astype(np.uint8), cv2.COLORMAP_INFERNO)))
                    else:
                        img_and_depth = np.hstack((cv2.cvtColor(cv2.resize(imgl, disparity.shape[::-1]),
                                                                cv2.COLOR_RGB2BGR), cv2.applyColorMap(
                            (disparity * 255 / disp_max).astype(np.uint8), cv2.COLORMAP_INFERNO)))
                    cv2.imshow("Input and disparity", img_and_depth)
                    cv2.waitKey(20)

    # Returning points
    # ret = []
    # if get_sift:
    #     ret.append(mse_sift)
    # if get_orb:
    #     ret.append(mse_orb)
    # if aru_core_pts:
    #     ret.append(mse_core)

    return core_es, [x_core, y_core], [x_pydnet, y_pydnet]


if __name__ == "__main__":
    # data_dir = "/home/kats/Datasets/Whitelab/Dataset_Structures/2022_07_04_10_26_41"
    data_dir = "/home/kats/Datasets/Route A/2022_07_06_10_48_24"
    dataset_obj = husky.DatasetHandler(data_dir, time_tolerance=0.5)
    # get_stereo_disp(dataset_obj.get_cam0(0), dataset_obj.get_cam1(0))
    max_frames = None

    # need to get the vo using the predicted disparity as well as using the disparity from stereo

    es15,  [x_core15, y_core15], [x_pydnet15, y_pydnet15] = predict_dataset_get_stereo_e(dataset=dataset_obj,scaling_factor=1,
                                                                               get_sift=False, aru_core_pts=True,
                                                                               get_orb=False, plots=False,
                                                                               max_frames=max_frames)

    es2,  [x_core2, y_core2], [x_pydnet2, y_pydnet2] = predict_dataset_get_stereo_e(dataset=dataset_obj,scaling_factor=2,
                                                                               get_sift=False, aru_core_pts=True,
                                                                               get_orb=False, plots=False,
                                                                               max_frames=max_frames)

    es25,  [x_core25, y_core25], [x_pydnet25, y_pydnet25] = predict_dataset_get_stereo_e(dataset=dataset_obj,scaling_factor=2.5,
                                                                               get_sift=False, aru_core_pts=True,
                                                                               get_orb=False, plots=False,
                                                                               max_frames=max_frames)

    # x_vo, y_vo = aru_vo_module.get_vo_path_on_dataset(dataset=dataset_obj)


    es1 = np.array(es1)
    mse1, me1, scale1 = es1.T

    es2 = np.array(es2)
    mse2, me2, scale2 = es2.T

    es25 = np.array(es25)
    mse25, me25, scale25 = es25.T

    plt.figure()
    plt.plot(mse1, label = "Scaling factor of 1")
    plt.plot(mse2, label = "Scaling factor of 2")
    plt.plot(mse25, label = "Scaling factor of 2.5")
    plt.title("MSE stereo disparity to pydnet disparity")
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(me1, label="Scaling factor of 1")
    plt.plot(me2, label = "Scaling factor of 2")
    plt.plot(me25, label = "Scaling factor of 2.5")
    plt.title("Mean Error stereo disparity to pydnet disparity")
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(scale1, label="Disparity scaled by factor of 1")
    plt.plot(scale2, label = "Disparity scaled by factor of 2")
    plt.plot(scale25, label = "Disparity scaled by factor of 2.5")
    plt.title("Scale stereo disparity / pydnet disparity")
    plt.legend()
    plt.show()

    plt.figure()
    # plt.plot(x_vo, y_vo, label="Bindings VO")
    plt.plot(x_core1, y_core1, label="Stereo VO")
    plt.plot(x_pydnet1, y_pydnet1, label="Pydnet VO scaled by 1")
    plt.plot(x_pydnet2, y_pydnet2, label="Pydnet VO scaled by 2")
    plt.plot(x_pydnet25, y_pydnet25, label="Pydnet VO scaled by 2.5")
    plt.legend()
    plt.axis('equal')
    plt.show()
    #
    # plt.plot(es)
    # plt.title(f"MSE between stereo and predicted depth by frame. Avg mse: {es.mean():.4f}")
    # plt.xlabel("Frame number")
    # plt.ylabel("MSE")
    # plt.show()
    # print("done")

# TODO check if theres a scale error between the left and the right disparity