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


def predict_dataset(dataset_dir=data_dir, save_fig=False, plots=True, verbose=False):
    dataset = get_dataset(dataset_dir)

    height = args.height
    width = args.width

    # Writing out images
    output_directory = os.path.join(data_dir, 'predictions')
    out_save_dir = os.path.join(output_directory, "output", 'image_00', 'data')
    query_save_dir = os.path.join(output_directory, "input", 'image_00', 'data')

    if not os.path.isdir(out_save_dir):
        os.makedirs(out_save_dir)
    if not os.path.isdir(query_save_dir):
        os.makedirs(query_save_dir)

    if plots:
        plt.ion()

        fig, [ax_l, ax_r] = plt.subplots(1, 2, figsize=(20, 10))
        fig.tight_layout(pad=3)

        l_title = ax_l.set_title("input image")
        ax_l.axis('off')
        r_title = ax_r.set_title("output depth image")
        ax_r.axis('off')

        cm = 'jet' # setting color map
        # cm = 'plasma' # setting color map


        plot_iml = ax_l.imshow(np.zeros((height, width), dtype=np.uint8))
        plot_imp = ax_r.imshow(np.zeros((height, width), dtype=np.uint8))


        divider = make_axes_locatable(ax_r)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cb = plt.colorbar(plot_imp, cax=cax)
        cb.set_label("Disparity [px]")



    with tf.Graph().as_default():
        placeholders, model, init, loader, saver = init_pydepth()
        show_flag = True

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            for counter in tqdm.trange(dataset.num_frames):
                # imgl_fname = dataset.get_cam0(counter)
                # imgr_fname = dataset.get_cam1(counter)
                # velo_fname = dataset.get_lidar(counter)
                imgl = cv2.cvtColor(dataset.get_cam0(counter), cv2.COLOR_BGR2RGB)

                img = cv2.resize(imgl, (width, height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)

                start_pred = time.time()
                disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end_pred = time.time()
                time_pred = start_pred - end_pred

                disparity = disp[0, :, :, 0].squeeze() * 0.3 * width

                depth = TR_func.disparity_to_depth(disparity, HuskyCalib.left_camera_matrix, HuskyCalib.t_cam0_velo)
                depth[depth > MAX_DISTANCE] = MAX_DISTANCE

                if plots:
                    plot_iml.set_data(imgl)
                    # plot_iml.set_data(cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB))

                    if "plot_imp" in locals():
                        plot_imp.remove()

                    plot_imp = ax_r.imshow(disparity)
                    plot_imp.set_cmap(cm)


                    cb.update_normal(plot_imp)

                    l_title.set_text(f"Input image for frame {counter}")
                    # r_title.set_text(f"Output depth map for frame {counter}")
                    r_title.set_text(f"Output disparity map for frame {counter}")


                    # only working in debug
                    # disparity not working

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                if save_fig:
                    out_save_path = os.path.join(out_save_dir, dataset.left_image_files[counter])
                    query_save_path = os.path.join(query_save_dir, dataset.left_image_files[counter])

                    if verbose: print(f"Saving to: {out_save_path}")
                    cv2.imwrite(out_save_path, depth)
                    cv2.imwrite(query_save_path, imgl)


if __name__ == "__main__":
    # Need to make tunable_recon() save images in an output directory

    RESAMPLING_ITERATIONS = 1
    predict_dataset(save_fig=False, plots=True)

