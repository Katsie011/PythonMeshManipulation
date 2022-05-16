import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
import sys

# from TunableReconstruction import IterativeTunableReconstructionPipeline

import ModularFiles.HuskyDataHandler as husky
import shutil

import tensorflow as tf
# import tensorflow.compat.v1 as tf
import argparse

from HyperParameters import *
from TunableReconstruction.ErrorEvaluation import *
from TunableReconstruction.Functions_TunableReconstruction import *
# from TunableReconstruction.IterativeTunableReconstructionPipeline import *
import ModularFiles.ImgFeatureExtactorModule as feat
import ModularFiles.HuskyDataHandler as husky

sys.path.insert(0, "./pydnet/")
from utils import *
from pydnet import *

# data_dir = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/"
# training_dir = "/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train1"
# test_dir = "/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test1"
# # test_dir = "/media/kats/Katsoulis3/Datasets/Husky/Training Data/TestKitti"
# test_filenames = os.path.join(test_dir, "test_file_list.txt")
# train_filenames = os.path.join(training_dir, 'training_file_list.txt')
# output_directory = '/media/kats/Katsoulis3/Datasets/Husky/Training Data/Test_Husky5000/'
# checkpoint_dir = '/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train1/tmp/Husky5000/Husky'

data_dir = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route C"
training_dir = "/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train_Route_C"
test_dir = "/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test_Route_C"
test_filenames = "/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test_Route_C/test_file_list.txt"
train_filenames = '/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train_Route_C/training_file_list.txt'
output_directory = os.path.join(test_dir, 'predictions')
# checkpoint_dir = '/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train1/tmp/Husky5000/Husky'
checkpoint_dir = '/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/mesh_pydnet/pydnet/checkpoint/Husky5000/Husky'
"""
 python experiments.py --datapath /media/kats/Katsoulis3/Datasets/Husky/Training\ Data/Test1/ \
 --filenames "/media/kats/Katsoulis3/Datasets/Husky/Training Data/Test1/test_file_list.txt" \
 --output_directory /media/kats/Katsoulis3/Datasets/Husky/Training\ Data/Test_Husky5000/
"""

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--dataset', type=str, help='dataset to train on, kitti, or Husky', default='Husky')
parser.add_argument('--datapath', type=str, help='path to the data', default=test_dir)  # required=True)
parser.add_argument('--filenames', type=str, help='path to the filenames text file',
                    default=test_filenames)  # required=True)
parser.add_argument('--output_directory', type=str,
                    help='output directory for test disparities, if empty outputs to checkpoint folder',
                    default=output_directory)
parser.add_argument('--checkpoint_dir', type=str, help='path to a specific checkpoint to load', default=checkpoint_dir)
parser.add_argument('--resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--save_predictions', type=bool, help="save predicted disparities to output directory",
                    default=True)
parser.add_argument('--width', dest='width', type=int, default=1280 // 2, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=768 // 2, help='height of input images')

args = parser.parse_args()


def read_image(image_path):
    # image  = tf.image.decode_image(tf.io.read_file(args.datapath+'/'+image_path))
    # image.set_shape( [None, None, 3])
    # image  = tf.image.convert_image_dtype(image,  tf.float32)
    # image  = tf.expand_dims(tf.image.resize(image,  [256, 512], tf.image.ResizeMethod.AREA), 0)

    image = cv2.imread(os.path.join(args.datapath, image_path))
    return image


def setup():
    dataset = tf.data.TextLineDataset(args.filenames)
    # If needing to batch the data
    # batches = dataset.batch(batch_size=10)
    # iterator = dataset.as_numpy_iterator()

    return dataset


def get_dataset():
    if args.dataset.lower() == 'husky':
        print("Using Husky data")
        # # print("\t Training Directory:", training_dir)
        # # files = pd.read_csv(train_filenames, sep=" ", names=["left", "right"]).sample(10).reset_index()
        #
        # print("\t Testing Directory:", test_dir)
        # left_files = os.listdir(os.path.join(test_dir, 'left'))
        # right_files = os.listdir(os.path.join(test_dir, 'right'))
        # files = pd.DataFrame(data=np.array((left_files, right_files)).T, columns=["left", "right"])

        return husky.Dataset_Handler(test_dir)

    elif args.dataset.lower() == 'kitti':
        print("Overriding and using Kitti Data")
        files = pd.DataFrame(data=np.array((data.cam2_files, data.cam3_files)).reshape((-1, 2)),
                             columns=['left', 'right'])
        files = files.sample(100).reset_index()

    else:
        raise "Invalid argument for dataset"
    return files


# def interpolate_pts(imgl, d_mesh, ft_uvd, verbose=False):
#     δ = cv2.Laplacian(imgl, cv2.CV_64F)
#     δ = np.abs(δ)
#
#     idx = np.argpartition(δ.flatten(), -INTERPOLATING_POINTS)[-INTERPOLATING_POINTS:]
#     gradient_pts = np.unravel_index(idx, imgl.shape)
#     interpolated_uv = np.stack((gradient_pts[1], gradient_pts[0]), axis=-1)
#     interpolated_pts = barycentric_interpolation(d_mesh, ft_uvd, interpolated_uv)
#     if verbose: print(f"Interpolated and returning {len(interpolated_pts)} points")
#     return interpolated_pts


def get_depth_pts(det, img, depth):
    u, v = keyPoint_to_UV(det.detect(img)).T
    u_d = np.round(u * depth.shape[1] / img.shape[1]).astype(int)
    v_d = np.round(v * depth.shape[0] / img.shape[0]).astype(int)
    d = depth[v_d, u_d]

    return np.stack((u, v, d), axis=-1)


def tunable_recon():
    dataset = get_dataset()

    det = feat.FeatureDetector(det_type='sift', max_num_ft=MAX_NUM_FEATURES_DETECT)

    # Writing out images
    out_save_dir = os.path.join(args.output_directory, "output", 'image_00', 'data')
    query_save_dir = os.path.join(args.output_directory, "input", 'image_00', 'data')

    if not os.path.isdir(out_save_dir):
        os.makedirs(out_save_dir)
    if not os.path.isdir(query_save_dir):
        os.makedirs(query_save_dir)

    # uv = keyPoint_to_UV(det.detect(img))

    with tf.Graph().as_default():
        height = args.height
        width = args.width
        placeholders = {'im0': tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='im0')}

        with tf.compat.v1.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer())

        loader = tf.compat.v1.train.Saver()
        saver = tf.compat.v1.train.Saver()

        show_flag = True

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            counter = 0
            for counter in range(dataset.num_frames):
                imgl_fname = dataset.left_image_files[counter]
                imgr_fname = dataset.right_image_files[counter]
                imgl = cv2.cvtColor(dataset.get_cam0(counter), cv2.COLOR_BGR2RGB)
                imgr = cv2.cvtColor(dataset.get_cam1(counter), cv2.COLOR_BGR2RGB)

                img = cv2.resize(imgl, (width, height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)

                start_pred = time.time()
                disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end_pred = time.time()
                time_pred = start_pred - end_pred

                disparity = disp[0, :, :, 0].squeeze() * 0.3 * width
                # depth = disparity_to_depth(np.round(disparity).astype(int), K, t)

                depth = disparity_to_depth(disparity, HuskyCalib.left_camera_matrix, HuskyCalib.t_cam0_velo)
                depth[depth > MAX_DISTANCE] = MAX_DISTANCE
                # depth = np.ma.array(depth, mask=depth > MAX_DISTANCE, fill_value=MAX_DISTANCE)

                it_start = time.time()
                mesh_pts = get_depth_pts(det, imgl, depth)
                depth_mesh = Delaunay(mesh_pts[:, :2])  # .filled())
                u, v, d = mesh_pts.T

                # ----------------------------------------------------
                #     Interpolating
                # ----------------------------------------------------

                to_resample = interpolate_pts(imgl, depth_mesh, mesh_pts, verbose=True)

                # ----------------------------------------------------
                #       Cost Calculation
                # ----------------------------------------------------

                c_interpolated, good_c_pts, bad_c_pts = calculate_costs(imgl, imgr, to_resample,
                                                                        mesh_pts, verbose=True)

                idx = np.argsort(bad_c_pts[:, -1])[::-1]
                num_pts_per_resample = 25
                eval_resampling_costs = False
                resampling_pts = bad_c_pts[idx[:num_pts_per_resample], :3]
                still_to_resample = bad_c_pts[idx[num_pts_per_resample:]]
                # cost_bad_pts = n x [u, v, d, c]
                resampled_pts = resample_iterate(imgl, imgr, resampling_pts,
                                                 eval_resampling_costs=eval_resampling_costs,
                                                 verbose=True)

                new_mesh_pts = np.vstack((mesh_pts, good_c_pts[:, :3], resampled_pts))
                new_mesh_pts = new_mesh_pts[new_mesh_pts[:, 2] < MAX_DISTANCE]
                new_mesh = Delaunay(new_mesh_pts[:, :2])  # .filled())

                # Now need to show a comparison before and after resampling
                it_stop = time.time()

                dpi = 40
                # figsize = 2*height / float(dpi), 2*width / float(dpi)
                figsize = width / float(dpi), height / float(dpi)
                fig, ax = plt.subplots(2, 3, figsize=figsize)  # , dpi=dpi)
                fig.tight_layout(pad=2)

                disp_im = ax[0, 0].imshow(disp[0, :, :, 0].squeeze(), 'jet')
                ax[0, 0].set_title(f"Disparity img frame {counter}")
                divider = make_axes_locatable(ax[0, 0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(disp_im, cax=cax, orientation='vertical')

                depth_im = ax[0, 1].imshow(depth, 'jet')
                ax[0, 1].set_title(f"Bounded depth img frame {counter}")
                divider = make_axes_locatable(ax[0, 1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(depth_im, cax=cax, orientation='vertical')

                ax[0, 2].hist(depth.flatten(), bins=100)
                ax[0, 2].set_title("Histogram of depth predictions")

                ax[1, 0].imshow(imgl)
                sc = ax[1, 0].scatter(u, v, c=d, s=10, cmap='jet')
                ax[1, 0].set_title(f"Scattered pts before Tunable Recon - Frame {counter}")
                divider = make_axes_locatable(ax[1, 0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(sc, cax=cax, orientation='vertical')

                ax[1, 1].imshow(imgl)
                u2, v2, d2 = new_mesh_pts.T
                sc1 = ax[1, 1].scatter(u2, v2, c=d2, s=10, cmap='jet')
                ax[1, 1].set_title(f"Scattered pts after Tunable Recon - Frame {counter}")
                divider = make_axes_locatable(ax[1, 1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(sc1, cax=cax, orientation='vertical')

                ax[1, 2].hist(new_mesh_pts[:, 2], bins=100)
                ax[1, 2].set_title("Histogram after tunable reconstruction")

                for axes in ax[:2, :2]:
                    for a in axes:
                        a.axis('off')

                # plt.show()
                fig.canvas.draw()
                plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                     sep='')
                plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                out_savepath = os.path.join(out_save_dir, dataset.left_image_files[counter])
                query_savepath = os.path.join(query_save_dir, dataset.left_image_files[counter])

                # img is rgb, convert to opencv's default bgr
                plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)

                print(f"Saving to: {out_savepath}")
                cv2.imwrite(out_savepath, depth)
                cv2.imwrite(query_savepath, imgl)
                cv2.imshow('Before and after', plot)
                cv2.waitKey(1)

                print(
                    f"Iteration:{counter}\n\tPred time: {time_pred}\n\ttunable_time{it_start - it_stop} for one "
                    f"iteration")
                # time.sleep(1)

                counter += 1


def predict_all():
    printflag = True

    data_files = get_dataset()
    imgs = []
    disps = []
    img_filenames = []

    with tf.Graph().as_default():

        height = args.height
        width = args.width
        placeholders = {'im0': tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='im0')}

        with tf.compat.v1.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer())

        loader = tf.compat.v1.train.Saver()
        saver = tf.compat.v1.train.Saver()

        show_flag = True

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            counter = 0
            while counter < len(data_files):
                if args.dataset.lower() == 'husky':
                    img_file = data_files['left'][counter]
                    img_path = os.path.join(test_dir, 'left', img_file)
                    # img_path = os.path.join(training_dir, img_file)
                else:
                    img_path = data_files['left'][counter]
                    img_file = f'left/{os.path.basename(img_path)}'
                # img_path = os.path.join(test_dir, data_files['left'][counter])
                img = read_image(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)
                start = time.time()
                disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end = time.time()

                disparity = disp[0, :, :, 0].squeeze()

                img_file = os.path.splitext(img_file)[0]

                imgs.append(img[0])
                img_filenames.append(img_file)
                disps.append(disparity)

                if show_flag:
                    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                    fig.tight_layout()
                    ax[0].imshow(imgs[-1])
                    ax[1].imshow(disps[-1], cmap='jet')
                    plt.show()

                    show_flag = False

                # if os.path.isdir(args.output_directory):
                #     # Saving to supplied directory
                #     if printflag:
                #         print(f"Saving to: {os.path.join(args.output_directory, img_path)}")
                #         printflag = True
                #
                #     np.save(os.path.join(args.output_directory, img_path), disparity)
                # else:
                #     # Saving to training data dir

                counter += 1

    # Writing out images
    out_save_dir = os.path.join(args.output_directory, "output")
    query_save_dir = os.path.join(args.output_directory, "input")

    if not os.path.isdir(os.path.split(os.path.join(out_save_dir, img_filenames[0]))[0]):
        os.makedirs(os.path.split(os.path.join(out_save_dir, img_filenames[0]))[0])
    if not os.path.isdir(os.path.split(os.path.join(query_save_dir, img_filenames[0]))[0]):
        os.makedirs(os.path.split(os.path.join(query_save_dir, img_filenames[0]))[0])

    for i in range(len(disps)):

        out_savepath = os.path.join(out_save_dir, img_filenames[i])
        query_savepath = os.path.join(query_save_dir, img_filenames[i])
        if printflag:
            print("Saving to:", out_savepath)
            printflag = False

        if not i % (len(data_files) // 10):
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            fig.tight_layout(pad=5.0)
            ax[0].imshow(imgs[i])
            ax[1].imshow(disps[i], cmap='jet')
            plt.show()

        # np.save(savepath+'.npy', disparity)

        # They scale disparity by 0.3*img width. So will multiply back by this factor

        cv2.imwrite(out_savepath + '.png', disps[i] * (0.3 * disparity.shape[1]))
        cv2.imwrite(query_savepath + '.png', imgs[i] * 255)
        cv2.imshow(f"Output {i}", imgs[i])

    pass


if __name__ == "__main__":
    # Need to make tunable_recon() save images in an output directory
    tunable_recon()

    # predict_all()
    pass
