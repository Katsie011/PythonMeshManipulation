import time

import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from scipy.spatial import Delaunay
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import argparse

from reconstruction.HyperParameters import *
import reconstruction.TunableReconstruction.Functions_TunableReconstruction as TR_func
# import reconstruction.ModularFiles.ImgFeatureExtactorModule as feat
import utilities.HuskyDataHandler as husky
# import reconstruction.mesh_pydnet.error_analytics.ErrorEvaluationImgs as ee



from pydnet.utils import *
from pydnet.pydnet import *

from skimage.metrics import structural_similarity as ssim




data_dir = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/RouteC/2022_05_03_14_09_01"
checkpoint_dir = '/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/mesh_pydnet/checkpoints/Husky10K/Husky'

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--dataset', type=str, help='dataset to train on, kitti, or Husky', default='Husky')
parser.add_argument('--datapath', type=str, help='path to the data', default=data_dir)  # required=True)
# parser.add_argument('--output_directory', type=str,
#                     help='output directory for test disparities, if empty outputs to checkpoint folder',
#                     default=output_directory)
parser.add_argument('--checkpoint_dir', type=str, help='path to a specific checkpoint to load', default=checkpoint_dir)
parser.add_argument('--resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--save_predictions', type=bool, help="save predicted disparities to output directory",
                    default=True)
parser.add_argument('--width', dest='width', type=int, default=1280, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=768, help='height of input images')

args = parser.parse_args()


def read_image(image_path):
    image = cv2.imread(os.path.join(args.datapath, image_path))
    return image


def get_dataset(dir=data_dir):
    if args.dataset.lower() == 'husky':
        print("Using Husky data")
        return husky.DatasetHandler(dir)


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
    print("not implemented")
    pass


def predict_dataset(dataset_dir=data_dir, save_fig=False, plots=True, verbose=False, training_size=(256,512)):
    dataset = get_dataset(dataset_dir)

    input_shape = dataset.img_shape
    full_width = int(128*(input_shape[1]//128))
    full_height = int(128*(input_shape[0]//128))

    # height = args.height
    # width = args.width

    train_height, train_width =training_size
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


    size_ssim = np.zeros(dataset.num_frames)
    size_mse = np.zeros(dataset.num_frames)

    if plots:
        plt.ion()

        fig, [[ax_l, ax_r], [ax_dis, ax_dep]] = plt.subplots(2, 2, figsize=(24, 14))
        fig.tight_layout(pad=5)

        l_title = ax_l.set_title("input image")
        ax_l.axis('off')
        r_title = ax_r.set_title("output inferred map")
        ax_r.axis('off')

        dis_title = ax_dis.set_title("inv_depth = inferred x 0.3 x width")
        ax_l.axis('off')
        dep_title = ax_dep.set_title("depth = (b x f)/inv_depth")
        ax_r.axis('off')

        cm = 'jet' # setting color map
        # cm = 'plasma' # setting color map


        plot_iml = ax_l.imshow(np.zeros((train_height, train_width), dtype=np.uint8))
        plot_imp = ax_r.imshow(np.zeros((train_height, train_width), dtype=np.uint8))

        plot_disp = ax_dis.imshow(np.zeros((train_height, train_width), dtype=np.uint8))
        plot_depth = ax_dep.imshow(np.zeros((train_height, train_width), dtype=np.uint8))

        divider = make_axes_locatable(ax_r)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb_imp = plt.colorbar(plot_disp, cax=cax)
        cb_imp.set_label("Disparity [px]")

        divider = make_axes_locatable(ax_dis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb_disp = plt.colorbar(plot_disp, cax=cax)
        cb_disp.set_label("Disparity [px]")

        divider = make_axes_locatable(ax_dep)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb_depth = plt.colorbar(plot_disp, cax=cax)
        cb_depth.set_label("Depth [m]")


    with tf.Graph().as_default():
        placeholders, model, init, loader, saver = init_pydepth()
        show_flag = True

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            for counter in tqdm.trange(dataset.num_frames):
                imgl = cv2.cvtColor(dataset.get_cam0(counter), cv2.COLOR_BGR2RGB)
                # img_full = cv2.resize(imgl, (full_width, full_height)).astype(np.float32)/255. # for comparing ssim with smaller prediction
                # img_full = np.expand_dims(img_full, 0)

                img = cv2.resize(imgl, (train_width, train_height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)

                # Getting training sized image depth
                start_pred = time.time()
                disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end_pred = time.time()
                time_pred = start_pred - end_pred
                disparity = disp[0, :, :, 0].squeeze() * 0.3 * train_width
                # disparity[disparity<MAX_DISPARITY] = MAX_DISPARITY
                depth = dataset.calib.baseline[0]*(dataset.calib.cam0_camera_matrix[0,0]*train_width/input_shape[1])/disparity
                depth[depth > MAX_DISTANCE] = MAX_DISTANCE
                if verbose:print(f"Time to predict image of size {training_size}: {time_pred}")



                # # predicting full sized image
                # start_pred_full = time.time()
                # disp_full = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img_full})
                # end_pred_full = time.time()
                # time_pred_full = start_pred_full - end_pred_full
                # disparity_full = disp_full[0, :, :, 0].squeeze() * 0.3 * full_width # using width of input shape this time
                # depth_full = dataset.calib.baseline[0]*(dataset.calib.cam0_camera_matrix[0,0]*full_width/input_shape[1])/disparity_full
                # depth_full[depth_full > MAX_DISTANCE] = MAX_DISTANCE
                # if verbose: print(f"Time to predict full size image of size {full_width, full_height}: {time_pred_full}")
                #
                #
                # # Getting difference in ssim for training depth and full sized image
                # upsized_depth = cv2.resize(depth, (full_width, full_height))
                # size_ssim[counter] = ssim(depth_full, upsized_depth)
                # size_mse[counter] = ((depth_full-upsized_depth)**2).mean()
                # print(f"SSIM for frame {counter}: \t{size_ssim[counter]:.3f}, MSE for frame {counter}: \t{size_mse[counter]:.3f}")
                #






                if plots:
                    plot_iml.set_data(imgl)
                    # plot_iml.set_data(cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB))

                    if "plot_imp" in locals():
                        plot_imp.remove()
                        plot_disp.remove()
                        plot_depth.remove()

                    # plot_imp = ax_r.imshow(disparity)
                    plot_imp = ax_r.imshow(disp[0, :, :, 0].squeeze())
                    plot_imp.set_cmap(cm)

                    # dis_title = ax_dis.set_title("inv_depth = inferred x 0.3 x width")
                    plot_disp = ax_dis.imshow(disparity)
                    plot_disp.set_cmap(cm)
                    plot_depth = ax_dep.imshow(depth)
                    # plot_depth.set_cmap(cm+"_r")
                    plot_depth.set_cmap(cm)

                    cb_imp.update_normal(plot_imp)
                    cb_disp.update_normal(plot_disp)
                    cb_depth.update_normal(plot_depth)

                    l_title.set_text(f"Input image for frame {counter}")
                    # r_title.set_text(f"Output depth map for frame {counter}")
                    r_title.set_text(f"Output inferred depth map for frame {counter}")


                    # only working in debug
                    # disparity not working

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                if save_fig:
                    out_save_path = os.path.join(out_save_dir, os.path.splitext(dataset.left_image_files[counter])[0])
                    disp_save_path = os.path.join(disp_save_dir, os.path.splitext(dataset.left_image_files[counter])[0])
                    query_save_path = os.path.join(query_save_dir, dataset.left_image_files[counter])

                    if verbose: print(f"Saving to: {out_save_path}")
                    np.save(out_save_path, cv2.resize(depth, (input_shape[1], input_shape[0])), allow_pickle=False)
                    np.save(disp_save_path, cv2.resize(disparity*(input_shape[1]/disparity.shape[1]), (input_shape[1], input_shape[0])), allow_pickle=False,)
                    cv2.imwrite(query_save_path, imgl)


    # return size_ssim, size_mse


if __name__ == "__main__":
    # Need to make tunable_recon() save images in an output directory
    # data_dir = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/RouteC/2022_05_03_14_09_01"
    dirs=["/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/BigBag/2022_06_22_14_23_58",
          "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route A/2022_05_03_13_53_38",
          "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route B/2022_05_04_09_38_30"]

    for data_dir in dirs:
        print(f"Predicting for bag:\n{data_dir}")
        predict_dataset(dataset_dir=data_dir, save_fig=True, plots=False)#, training_size=(args.height, args.width))




    # # From when comparing different input sizes
    # ssims, mses = predict_dataset(save_fig=False, plots=False, training_size=(args.height, args.width))
    #
    # plt.figure()
    # plt.plot(ssims)
    # plt.title("SSIMs between full sized predictions and training sized predictions")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(mses)
    # plt.title("MSEs between full sized predictions and training sized predictions")
    # plt.show()

