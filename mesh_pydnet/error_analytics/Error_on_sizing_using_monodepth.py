"""
Getting the error metrics for input images using the monodepth framework.

Need to check whether the resizing of output and conversion to depth is correct.
"""

import numpy as np
import argparse
import re
import time
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# import tensorflow.contrib.slim as slim
import tqdm
import utilities.HuskyDataHandler as husky
import matplotlib.pyplot as plt
import os
import cv2

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as img_mse

import sys

sys.path.append("/home/kats/Code/aru_sil_py/reconstruction/mesh_pydnet/pydnet/training_code/tf2_converted")

from reconstruction.mesh_pydnet.pydnet.training_code.tf2_converted.monodepth_model import *
from reconstruction.mesh_pydnet.pydnet.training_code.tf2_converted.monodepth_dataloader import *
from reconstruction.mesh_pydnet.pydnet.training_code.tf2_converted.average_gradients import *



from reconstruction.HyperParameters import *
import utilities.HuskyDataHandler as husky
import reconstruction.VOComparison.aru_visual_odometry as vocomp
import utilities.ImgFeatureExtactorModule as ft
sift = ft.FeatureDetector(det_type='sift', max_num_ft=2000)
orb = ft.FeatureDetector(det_type='orb', max_num_ft=2000)


parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder', type=str, help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset_path', type=str, help='path to the husky dataset',
                    default=r"/home/kats/Datasets/Route A/2022_07_06_10_48_24/")
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load',
                    default=r"/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/"
                            r"mesh_pydnet/checkpoints/Husky10K/Husky")
parser.add_argument('--input_height', type=int, help='input height', default=256)
parser.add_argument('--input_width', type=int, help='input width', default=512)

args = parser.parse_args()


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def run_simple(params):
    """simple run function."""
    train_width = args.input_width
    train_height = args.input_height
    MAX_DISTANCE = 100
    get_sift = True
    get_orb=False
    max_frames=None
    dataset = husky.DatasetHandler(args.dataset_path)
    plots = True

    # setiing up vo


    if get_sift:
        sift_coords = np.eye(4)
    if get_orb:
        orb_coords = np.eye(4)

    if max_frames is None:
        max_frames = dataset.num_frames - 1
    if get_sift:
        x_sift, y_sift = np.zeros(max_frames), np.zeros(max_frames)
    if get_orb:
        x_orb, y_orb = np.zeros(max_frames), np.zeros(max_frames)


    with tf.Graph().as_default():
        left = tf.compat.v1.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
        right = tf.compat.v1.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
        # model = MonodepthModel(params, "test", left, right)
        print("Creating model in training mode:")
        model = MonodepthModel(params, "train", left, right)
        print("\t Done.")
        frame_num = 0
        # input_image = scipy.misc.imread(args.image_path, mode="RGB")
        left_input_image = dataset.get_cam0(frame_num)
        original_height, original_width, num_channels = left_input_image.shape

        # SESSION
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        sess = tf.compat.v1.Session(config=config)

        # SAVER
        train_saver = tf.compat.v1.train.Saver()

        # INIT
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coordinator)

        # RESTORE
        restore_path = args.checkpoint_path.split(".")[0]
        train_saver.restore(sess, restore_path)

        # PREDICT
        for counter in tqdm.trange(dataset.num_frames - 1):
            imgl = dataset.get_cam0(counter)
            left_input_image = cv2.resize(imgl, (args.input_width, args.input_height))
            left_smaller = left_input_image
            left_input_image = left_input_image.astype(np.float32) / 255
            left_input_images = np.stack((left_input_image, np.fliplr(left_input_image)), 0)

            imgr = dataset.get_cam1(counter)
            right_input_image = cv2.resize(imgr, (args.input_width, args.input_height))
            right_input_image = right_input_image.astype(np.float32) / 255
            right_input_images = np.stack((right_input_image, np.fliplr(right_input_image)), 0)

            disp = sess.run(model.disp_left_est[0],
                            feed_dict={left: left_input_images, right: right_input_images}).squeeze()

            disp_pp = post_process_disparity(disp).astype(np.float32)
            l_disp = disp[0, :, :]
            disparity_pp = disp_pp * 0.3 * disp_pp.shape[1]

            # disparity = disp[0, :, :, 0].squeeze() * 0.3 * train_width
            # disparity[disparity<MAX_DISPARITY] = MAX_DISPARITY
            depth_pp = dataset.calib.baseline[0] * (
                    dataset.calib.cam0_camera_matrix[0, 0] * disparity_pp.shape[1] / dataset.img_shape[
                1]) / disparity_pp
            depth_pp[depth_pp > MAX_DISTANCE] = MAX_DISTANCE
            fullsize_depth = cv2.resize(depth_pp, (dataset.img_shape[1], dataset.img_shape[0]))

            # Getting difference metrics:

            big_disp = cv2.resize(disp_pp, (dataset.img_shape[1], dataset.img_shape[0])) * 0.3 * dataset.img_shape[1]
            big_depth = dataset.calib.baseline[0] * (
                        dataset.calib.cam0_camera_matrix[0, 0] * big_disp.shape[1] / dataset.img_shape[1]) / big_disp
            big_depth[big_depth>MAX_DISTANCE] = MAX_DISTANCE
            depth_resized_small = cv2.resize(big_depth, disp_pp.shape[:2][::-1])
            disparity_resized_small = dataset.calib.baseline[0] * (dataset.calib.cam0_camera_matrix[0, 0] *
                                                              depth_resized_small.shape[1] / dataset.img_shape[
                                                                  1]) / depth_resized_small
            disp_resized_small = disparity_resized_small/(0.3*disparity_resized_small.shape[1])


            assert l_disp.shape == disp_pp.shape, "left disparity shape does not match post process disparity shape"
            ssim_l_disp_to_disp_pp = ssim(l_disp, disp_pp)
            ssim_disp_small_pp_to_big_pp = ssim(l_disp, disp_resized_small)

            print(f"Comparison of resizing metrics:\n"
                  f"\tsmall disparity to big depth --> disparity resized to small: {ssim_disp_small_pp_to_big_pp:.5f}\n"
                  f"\tleft disparity to post processed disparity output: \t{ssim_l_disp_to_disp_pp}")


            # cv2.imshow("small and big resized small", np.hstack((
            #     cv2.applyColorMap((l_disp * 255 / l_disp.max()).astype(np.uint8), cv2.COLORMAP_PLASMA),
            #     cv2.applyColorMap((disp_resized_small * 255 / disp_resized_small.max()).astype(np.uint8),
            #                       cv2.COLORMAP_PLASMA))))



            # DOING VO
            imgl_p1 = dataset.get_cam0(counter + 1)
            if get_orb:
                orb_transform = vocomp.predicted_depth_motion_est(depth_f0=fullsize_depth, img_f0=imgl,
                                                                  img_f1=imgl_p1,
                                                                  K=dataset.calib.cam0_camera_matrix, det=orb)
                # if plots:
                #     kps0, des0 = orb.detect(imgl)
                #     kps1, des1 = orb.detect(imgl_p1)
                #     matches = orb.get_matches(des0, des1, lowes_ratio=0.7)
                #     # kps0_img.set_data(cv2.drawKeypoints(imgl, kps0, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
                #     # kps1_img.set_data(cv2.drawKeypoints(imgl_p1, kps1, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
                #     img_kps0 = cv2.drawKeypoints(imgl, kps0, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                #     img_kps1 = cv2.drawKeypoints(imgl_p1, kps1, 0, (0, 255, 0),
                #                                  flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                #
                #     cv2.imshow("ORBKeypointsImg",
                #                cv2.resize(np.hstack((img_kps0, img_kps1)), (dataset.img_shape[1], dataset.img_shape[0] // 2)))

                if orb_transform is not None:
                    orb_coords = orb_coords @ orb_transform
                    x_orb[counter] = -orb_coords[2][3]
                    y_orb[counter] = orb_coords[0][3]
                else:
                    print(f"None returned using ORB for depth VO in frame {counter}")

            if get_sift:
                sift_transform = vocomp.predicted_depth_motion_est(depth_f0=fullsize_depth, img_f0=imgl,
                                                                   img_f1=imgl_p1,
                                                                   K=dataset.calib.cam0_camera_matrix, det=sift)

                # if plots:
                #     kps0, des0 = sift.detect(imgl)
                #     kps1, des1 = sift.detect(imgl_p1)
                #     matches = sift.get_matches(des0, des1, lowes_ratio=0.7)
                #     # kps0_img.set_data(cv2.drawKeypoints(imgl, kps0, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
                #     # kps1_img.set_data(cv2.drawKeypoints(imgl_p1, kps1, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
                #     img_kps0 = cv2.drawKeypoints(imgl, kps0, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                #     img_kps1 = cv2.drawKeypoints(imgl_p1, kps1, 0, (0, 255, 0),
                #                                  flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                #
                #     cv2.imshow("SIFTKeypointsImg",
                #                cv2.resize(np.hstack((img_kps0, img_kps1)), (dataset.img_shape[1], dataset.img_shape[0] // 2)))

                if sift_transform is not None:
                    sift_coords = sift_coords @ sift_transform
                    x_sift[counter] = -sift_coords[2][3]
                    y_sift[counter] = sift_coords[0][3]
                else:
                    print(f"None returned using SIFT for depth VO in frame {counter}")

            # PLOTS

            # cv2.imshow("Input image", np.hstack((input_images[0], input_images[1])))
            # cv2.imshow("Predicted_image", cv2.applyColorMap((255*depth/depth.max()).astype(np.uint8), cv2.COLORMAP_JET))
            # plt_depth = cv2.resize(plt_depth, (train_width, train_height))
            # cv2.imshow("Input image and depth", np.hstack((left_smaller, cv2.applyColorMap(
            #     ( depth* 255 / MAX_DISTANCE).round().astype(np.uint8), cv2.COLORMAP_PLASMA_R))))

            plt_disp = cv2.resize(disparity_pp, (train_width, train_height)) * train_width / disparity_pp.shape[1]

            cv2.imshow("Input image and depth", np.hstack((left_smaller, cv2.applyColorMap(
                (plt_disp * 255 / plt_disp.max()).astype(np.uint8), cv2.COLORMAP_PLASMA))))
            cv2.waitKey(2)

            print(model.disp_gradient_loss)

        print('done!')

    # Returning points
    if get_sift and get_orb:
        return [x_orb, y_orb], [x_sift, y_sift]
    elif get_sift and not get_orb:
        return [x_sift, y_sift]
    elif get_orb:
        return [y_orb, y_orb]


def main(_):
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    [x_s, y_s] = run_simple(params)
    dataset_obj = husky.DatasetHandler(args.dataset_path)
    x_vo, y_vo = vocomp.get_vo_path_on_dataset(dataset=dataset_obj)

    import matplotlib.pyplot as plt
    import utilities.plotting_utils as VO_plt

    fig, ax = plt.subplots()
    # VO_plt.plot_vo_path_with_arrows(axis=ax, x=x_o, y=-y_o, linestyle='o--', label="Pydnet ORB VO")
    VO_plt.plot_vo_path_with_arrows(axis=ax, x=x_s, y=-y_s, linestyle='o--', label="Pydnet SIFT VO")
    VO_plt.plot_vo_path_with_arrows(axis=ax, x=x_vo, y=y_vo, linestyle='o--', label="Stereo VO")
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main("")


