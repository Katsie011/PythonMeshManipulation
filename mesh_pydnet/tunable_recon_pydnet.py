import time

import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from scipy.spatial import Delaunay
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import argparse

from mesh_pydnet.HyperParameters import *
import TunableReconstruction.IterativeTunableReconstructionPipeline as ITR
import TunableReconstruction.Functions_TunableReconstruction as TR_func
import ModularFiles.ImgFeatureExtactorModule as feat
import ModularFiles.HuskyDataHandler as husky
import TunableReconstruction.ErrorEvaluationImgs as ee

from pydnet.utils import *
from pydnet.pydnet import *

data_dir = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route C"
training_dir = "/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train_Route_C"
test_dir = "/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test_Route_C"
test_filenames = "/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test_Route_C/test_file_list.txt"
train_filenames = '/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train_Route_C/training_file_list.txt'
output_directory = os.path.join(test_dir, 'predictions')
# checkpoint_dir = '/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train1/tmp/Husky5000/Husky'
checkpoint_dir = '/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/mesh_pydnet/pydnet/checkpoint/Husky5000/Husky'

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

    # elif args.dataset.lower() == 'kitti':
    #     print("Overriding and using Kitti Data")
    #     files = pd.DataFrame(data=np.array((data.cam2_files, data.cam3_files)).reshape((-1, 2)),
    #                          columns=['left', 'right'])
    #     files = files.sample(100).reset_index()
    #
    # else:
    #     raise "Invalid argument for dataset"
    # return files


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
    query_im = dataset.get_cam0(frame)
    velo = dataset.get_lidar(frame)
    u, v, gt_depth = ee.lidar_to_img_frame(velo, HuskyCalib.T_cam0_vel0, dataset.calib.cam0_camera_matrix,
                                           img_shape=query_im.shape)
    gt_disparity = TR_func.depth_to_disparity(gt_depth, dataset.calib.cam0_camera_matrix, dataset.calib.baseline)
    velo_cam = np.floor(np.stack((u, v, gt_disparity), axis=1)).astype(int)

    pred_e = ee.quantify_img_error_lidar(pred_disp_img, velo_cam)

    errors = np.zeros(len(pt_list))
    if plot:
        # if make plot of pts:
        cols = 2
        rows = len(pt_list)+1
        fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
        fig.tight_layout(pad=2)

        ax[0, 0].imshow(query_im)
        ax[0, 0].set_title(f"Query image for frame {frame}")
        ax[0, 0].axis('off')

        im = ax[0, 1].imshow(pred_disp_img, 'jet')
        ax[0, 1].set_title("Predicted Disparity")
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Disparity")
        ax[0, 1].axis('off')

    # make lidar into ground truth image
    gt_im = ee.rough_lidar_render(velo_cam)

    for i, pts in enumerate(pt_list):
        # get predicted depth for each point from the pts
        # u, v, samples = ee.sample_img(pred_disp_img, pts[:, :2])
        u, v, samples = pts.T
        # Calculate error w.r.t. lidar gt img
        err = ee.get_img_pt_to_pt_error(gt_im, np.stack((u, v), axis=1), samples,
                                        img_shape=query_im.shape, use_MSE=True)

        # append to errors list
        errors[i] = err

        if plot:
            print(f"Using axis: [{1 // cols}, {i % cols}]")
            a = ax[1 + i, 0]
            a.axis('off')
            a.imshow(query_im)
            # Make Mesh
            mesh = Delaunay(pts[:, :2])

            # Triplot of mesh
            # TODO

            # color scattered points by their error
            sc = a.scatter(u, v, c=samples, s=5, cmap='jet')
            divider = make_axes_locatable(a)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(sc, cax=cax)
            cb.set_label("Disparity")
            a.set_title(title_list[i] + f" - MSE: {err:.1f} and with {len(pt_list[i])} pts")

            b = ax[1 + i, 1]
            b.axis('off')

            u, v, gt_depth = ee.lidar_to_img_frame(velo, HuskyCalib.T_cam0_vel0, dataset.calib.cam0_camera_matrix,
                                                img_shape=query_im.shape)
            gt_disparity = TR_func.depth_to_disparity(gt_depth, dataset.calib.cam0_camera_matrix, dataset.calib.baseline)
            velo_cam = np.floor(np.stack((u, v, gt_disparity), axis=1)).astype(int)
            im = b.imshow(ee.pt_to_pt_emap(predicted_img=pred_disp_img, gt_cam_pts=velo_cam, mask=True,
                                                  colourmap=None), 'jet')
            divider = make_axes_locatable(b)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label("Disparity MSE")
            b.set_title(f"Error Map. MSE:{errors[i]:.2f}")


    if plot:
        fig.canvas.draw()
        plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                             sep='')
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return errors, plot

    return errors


def tunable_recon(save_fig=False, verbose=False):
    dataset = get_dataset()

    sift = feat.FeatureDetector(det_type='sift', max_num_ft=MAX_NUM_FEATURES_DETECT)
    orb = feat.FeatureDetector(det_type='orb', max_num_ft=MAX_NUM_FEATURES_DETECT)
    # surf = feat.FeatureDetector(det_type='surf', max_num_ft=MAX_NUM_FEATURES_DETECT)

    e_sift = []
    e_orb = []

    # Writing out images
    out_save_dir = os.path.join(args.output_directory, "output", 'image_00', 'data')
    query_save_dir = os.path.join(args.output_directory, "input", 'image_00', 'data')

    if not os.path.isdir(out_save_dir):
        os.makedirs(out_save_dir)
    if not os.path.isdir(query_save_dir):
        os.makedirs(query_save_dir)

    with tf.Graph().as_default():
        height = args.height
        width = args.width
        placeholders, model, init, loader, saver = init_pydepth()
        show_flag = True

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            for counter in tqdm.trange(dataset.num_frames):
                imgl_fname = dataset.get_cam0(counter)
                imgr_fname = dataset.get_cam1(counter)
                velo_fname = dataset.get_lidar(counter)
                imgl = dataset.get_cam0(counter)
                imgr = dataset.get_cam1(counter)

                img = cv2.resize(imgl, (width, height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)

                start_pred = time.time()
                disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end_pred = time.time()
                time_pred = start_pred - end_pred

                disparity = disp[0, :, :, 0].squeeze() * 0.3 * width

                depth = TR_func.disparity_to_depth(disparity, HuskyCalib.left_camera_matrix, HuskyCalib.t_cam0_velo)
                depth[depth > MAX_DISTANCE] = MAX_DISTANCE
                # depth = np.ma.array(depth, mask=depth > MAX_DISTANCE, fill_value=MAX_DISTANCE)

                # it_start = time.time()
                # itr_points = ITR.depth_img_iterative_recon(img_l=imgl, img_r=imgr, depth_prediction=depth,
                #                                            return_plot=False, frame_num=counter, img_shape=imgl.shape,
                #                                            verbose=False)
                # itr_mesh = Delaunay(itr_points[:, :2])
                # Now need to show a comparison before and after resampling
                # it_stop = time.time()

                orb_pts = TR_func.get_depth_pts(orb, imgl, disparity)
                sift_pts = TR_func.get_depth_pts(sift, imgl, disparity)
                # surf_pts = TR_func.get_depth_pts(surf, imgl, disparity)

                pt_list = [orb_pts, sift_pts]  # , surf_pts]
                title_list = ["Orb Points", "Sift Points"]  # , "Surf Points"]

                errors, plot = get_errors(disparity, dataset, frame=counter, pt_list=pt_list, title_list=title_list,
                                          plot=True)

                e_orb.append(errors[0])
                e_sift.append(errors[1])


                # img is rgb, convert to opencv's default bgr
                plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
                cv2.imshow('Before and after', plot)
                cv2.waitKey(1)

                e_fig, e_ax = plt.subplots()
                e_fig.suptitle("MSE Errors by frame")

                e_ax.plot(e_orb, label="ORB MSE")
                e_ax.plot(e_sift, label="SIFT MSE")

                e_ax.legend(loc='upper left')

                e_fig.canvas.draw()
                canv = np.fromstring(e_fig.canvas.tostring_rgb(), dtype=np.uint8,
                                     sep='')
                canv = canv.reshape(e_fig.canvas.get_width_height()[::-1] + (3,))
                canv = cv2.cvtColor(canv, cv2.COLOR_RGB2BGR)
                cv2.imshow('Error Plots', canv)
                cv2.waitKey(1)

                print(f"Avg Errors\n"
                      f"\tORB: {np.mean(e_orb)}\n"
                      f"\tSIFT: {np.mean(e_sift)}\n\n")

                if save_fig:
                    out_save_path = os.path.join(out_save_dir, dataset.left_image_files[counter])
                    query_save_path = os.path.join(query_save_dir, dataset.left_image_files[counter])

                    print(f"Saving to: {out_save_path}")
                    cv2.imwrite(out_save_path, depth)
                    cv2.imwrite(query_save_path, imgl)

                # print(
                #     f"Iteration:{counter}\n\tPred time: {time_pred}\n\ttunable_time{it_start - it_stop} for one "
                #     f"iteration")
                # time.sleep(1)


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

    RESAMPLING_ITERATIONS = 1
    tunable_recon()

    # predict_all()
    pass
