"""
Online VO using Pydnet network
"""
import time
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
# from scipy.spatial import Delaunay
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import argparse

from reconstruction.HyperParameters import *
import utilities.HuskyDataHandler as husky
import reconstruction.VOComparison.aru_visual_odometry as vocomp
import utilities.ImgFeatureExtactorModule as ft

from pydnet.utils import *
from pydnet.pydnet import *

sift = ft.FeatureDetector(det_type='sift', max_num_ft=2000)
orb = ft.FeatureDetector(det_type='orb', max_num_ft=2000)

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


def read_image(image_path):
    return cv2.imread(os.path.join(args.datapath, image_path))


def get_dataset(dir=args.datapath):
    if args.dataset.lower() == 'husky':
        print("Using Husky data")
        return husky.DatasetHandler(dir)
    else:
        print("Datasets other than for the Husky are not supported yet")
        raise NotImplementedError


def init_pydepth():
    placeholders = {'im0': tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='im0')}

    with tf.compat.v1.variable_scope("model") as scope:
        # pydnet might look like an error but it depends on if your IDE can load the module from a sys include
        model = pydnet(placeholders)

    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())

    loader = tf.compat.v1.train.Saver()
    saver = tf.compat.v1.train.Saver()

    return placeholders, model, init, loader, saver


def get_errors(pred_disp_img, dataset, frame, pt_list, title_list, mse=True, plot=False):
    """
    Gets the error of the prediction compared to the lidar as well as the pt-to-pt error for the points provided

    If plot == true:
        makes a plot of the errors and displays
    """
    raise NotImplementedError


def predict_dataset(dataset, plots=True, verbose=False, training_size=(256, 512), max_frames=None, get_orb=True,
                    get_sift=True):
    # dataset = get_dataset(dataset_dir)

    if not get_orb and not get_sift:
        print("No feature detector chosen. Returning None")
        return None

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

    input_shape = dataset.img_shape
    full_width = int(128 * (input_shape[1] // 128))
    full_height = int(128 * (input_shape[0] // 128))

    train_height, train_width = training_size
    # Writing out images
    output_directory = os.path.join(data_dir, 'predictions')
    out_save_dir = os.path.join(output_directory, "output", 'image_00', 'data')
    disp_save_dir = os.path.join(output_directory, "disparity", 'image_00', 'data')
    query_save_dir = os.path.join(output_directory, "input", 'image_00', 'data')

    if not os.path.isdir(out_save_dir):
        os.makedirs(out_save_dir)
    if not os.path.isdir(query_save_dir):
        os.makedirs(query_save_dir)
    if not os.path.isdir(disp_save_dir):
        os.makedirs(disp_save_dir)

    # if plots:
    #     plt.ion()
    #
    #     fig, [[ax_l, ax_r], [ax_kps0, ax_kps1]] = plt.subplots(2, 2, figsize=(24, 14))
    #     fig.tight_layout(pad=5)
    #     l_title = ax_l.set_title("input image")
    #     plt_input_img = ax_l.imshow(np.zeros(input_shape))
    #     ax_l.axis('off')
    #     r_title = ax_r.set_title("output inferred map")
    #     plt_onput_img = ax_r.imshow(np.zeros(input_shape), 'jet')
    #     ax_r.axis('off')
    #     kps0_img = ax_kps0.imshow(np.zeros(input_shape))
    #     ax_kps0.set_title("Keypoints on frame n")
    #     ax_kps0.axis('off')
    #     kps1_img = ax_kps1.imshow(np.zeros(input_shape))
    #     ax_kps1.set_title("Keypoints on frame n+1")
    #     ax_kps1.axis('off')
    #
    #     # plt.show()




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
                depth = dataset.calib.baseline[0] * (dataset.calib.cam0_camera_matrix[0, 0] * train_width / input_shape[1]) / disparity
                depth[depth > MAX_DISTANCE] = MAX_DISTANCE
                fullsize_depth = cv2.resize(depth, (full_width, full_height))
                if verbose: print(f"Time to predict image of size {training_size}: {time_pred}")

                # if plots:
                #     plt_input_img.set_data(imgl)
                #     l_title.set_text(f"Input image for frame {counter}")
                #     plt_onput_img.set_data(fullsize_depth)
                #     r_title.set_text(f"Fullsize Depth prediction frame {counter}")

                ########################################################################################################
                #                                           Getting the VO                                             #
                ########################################################################################################
                imgl_p1 = cv2.cvtColor(dataset.get_cam0(counter + 1), cv2.COLOR_BGR2RGB)
                if get_orb:
                    orb_transform = vocomp.predicted_depth_motion_est(depth_f0=fullsize_depth, img_f0=imgl,
                                                                      img_f1=imgl_p1,
                                                                      K=dataset.calib.cam0_camera_matrix, det=orb)
                    if plots:
                        kps0, des0 = orb.detect(imgl)
                        kps1, des1 = orb.detect(imgl_p1)
                        matches = orb.get_matches(des0, des1, lowes_ratio=0.7)
                        # kps0_img.set_data(cv2.drawKeypoints(imgl, kps0, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
                        # kps1_img.set_data(cv2.drawKeypoints(imgl_p1, kps1, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
                        img_kps0= cv2.drawKeypoints(imgl, kps0, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                        img_kps1= cv2.drawKeypoints(imgl_p1, kps1, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

                        cv2.imshow("ORBKeypointsImg", cv2.resize(np.hstack((img_kps0, img_kps1)), (full_width, full_height//2)))



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

                    if plots:
                        kps0, des0 = sift.detect(imgl)
                        kps1, des1 = sift.detect(imgl_p1)
                        matches = sift.get_matches(des0, des1, lowes_ratio=0.7)
                        # kps0_img.set_data(cv2.drawKeypoints(imgl, kps0, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
                        # kps1_img.set_data(cv2.drawKeypoints(imgl_p1, kps1, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
                        img_kps0 = cv2.drawKeypoints(imgl, kps0, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                        img_kps1 = cv2.drawKeypoints(imgl_p1, kps1, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

                        cv2.imshow("SIFTKeypointsImg",
                                   cv2.resize(np.hstack((img_kps0, img_kps1)), (full_width, full_height // 2)))

                    if sift_transform is not None:
                        sift_coords = sift_coords @ sift_transform
                        x_sift[counter] = -sift_coords[2][3]
                        y_sift[counter] = sift_coords[0][3]
                    else:
                        print(f"None returned using SIFT for depth VO in frame {counter}")

                end_loop_time = time.time()
                print(
                    f"\nTook {end_loop_time - start_loop_time:.3f}s to run one iteration with SIFT and ORB VO together")

                if plots:
                    cv2.waitKey(20)

    # Returning points
    if get_sift and get_orb:
        return [x_orb, y_orb], [x_sift, y_sift]
    elif get_sift and not get_orb:
        return [x_sift, y_sift]
    elif get_orb:
        return [y_orb, y_orb]


if __name__ == "__main__":
    # Need to make tunable_recon() save images in an output directory
    # data_dir = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/RouteC/2022_05_03_14_09_01"
    # dirs=["/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/BigBag/2022_06_22_14_23_58",
    #       "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route A/2022_05_03_13_53_38",
    #       "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route B/2022_05_04_09_38_30"]

    data_dir = "/home/kats/Datasets/Whitelab/Dataset_Structures/2022_07_04_10_26_41"
    dataset_obj = husky.DatasetHandler(data_dir, time_tolerance=0.5)
    print(f"Predicting for bag:\n{data_dir}")
    num_frames = None

    [x_o, y_o], [x_s, y_s] = predict_dataset(dataset=dataset_obj, plots=True, max_frames=num_frames)
    # [x_o, y_o] = predict_dataset(dataset=dataset_obj, plots=False, max_frames=num_frames, get_orb=True, get_sift=False)
    # [x_s, y_s] = predict_dataset(dataset=dataset_obj, plots=True, max_frames=num_frames, get_orb=False, get_sift=True)
    x_vo, y_vo = vocomp.get_vo_path_on_dataset(dataset=dataset_obj, stop_frame=num_frames)

    print("Plotting")
    import matplotlib.pyplot as plt
    import utilities.plotting_utils as VO_plt

    fig, ax = plt.subplots()
    VO_plt.plot_vo_path_with_arrows(axis=ax, x=x_o, y=-y_o, linestyle='o--', label="Pydnet ORB VO")
    VO_plt.plot_vo_path_with_arrows(axis=ax, x=x_s, y=-y_s, linestyle='o--', label="Pydnet SIFT VO")
    VO_plt.plot_vo_path_with_arrows(axis=ax, x=x_vo, y=y_vo, linestyle='o--', label="Stereo VO")
    ax.legend()

    plt.show()
