"""

Goal of this file is to do all the mesh error evaluation in one place
This error evaluation was done previously in notebook files
Want one package that can quickly determine error and give quantitative results

Options:
    - Quantify error
        - MSE
    - Generate error heat maps
    - Create error plots for several images in a stream.
        - Option to make videos from plots

"""

# ----------------------------------------------------------------------------
#       Imports
# ----------------------------------------------------------------------------
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys

import numpy as np

from PythonMeshManipulation.mesh_pydnet.HyperParameters import *
# from PythonMeshManipulation.mesh_pydnet.TunableReconstruction.Functions_TunableReconstruction import *
import ModularFiles.HuskyDataHandler as husky
import IterativeTunableReconstructionPipeline as ITR

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
#       Globals
# ----------------------------------------------------------------------------

# config_path = "/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/Tunable " \
# "Reconstruction/mesh_depth.yaml "
config_path = "/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/mesh_pydnet" \
              "/TunableReconstruction/mesh_depth.yaml"

depth_est = aru_py_mesh.PyDepth(config_path)


def make_sparse_dense(sparse_img, get_colour=False, mask=True):
    # colour, depth = depth_est.create_dense_depth(sparse_img)
    colour, depth = depth_est.create_dense_depth(sparse_img)

    if mask:
        depth = np.ma.array(depth, mask=(depth == -1), fill_value=None)
        if get_colour:
            colour = np.ma.array(colour, mask=(colour == -1), fill_value=None)

    if get_colour:
        return colour, depth
    else:
        return depth


def pts_to_img(pts, imshape=IMAGE_SHAPE, dtype=np.uint8):
    im = np.zeros(imshape, dtype=dtype)
    idx = np.floor(pts[:, :2].T).astype(int)
    im[idx[1], idx[0]] = pts[:, 2]
    return im


def lidar_to_img_frame(pts, Tr=HuskyCalib.T_cam0_vel0, K=HuskyCalib.left_camera_matrix, img_shape=IMAGE_SHAPE):
    pts = np.hstack((pts, np.ones((len(pts), 1))))
    homogeneous = np.matmul(Tr, pts.T)
    u = (K[0, 0] * homogeneous[0] / homogeneous[2]) + K[0, 2]
    v = (K[1, 1] * homogeneous[1] / homogeneous[2]) + K[1, 2]
    d = homogeneous[2]

    idx = (v > 0) & (v < img_shape[0]) & (u > 0) & (u < img_shape[1]) & (d > 0)
    u, v, d = u[idx], v[idx], d[idx]

    return u, v, d


def render_lidar(pts, Tr=HuskyCalib.T_cam0_vel0, K=HuskyCalib.left_camera_matrix, img_shape=IMAGE_SHAPE,
                 render_shape=PREDICTION_SHAPE, mask=True):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
       numpy array. Returns the 2D projection of the points that
       are in front of the camera only an the corresponding 3D points.'''

    if len(img_shape) == 3:
        img_shape = img_shape[:2]

    if len(render_shape) == 3:
        render_shape = render_shape[:2]

    # u, v, d = lidar_to_img_frame(pts, Tr, K, img_shape=img_shape)
    u, v, d = pts.T

    lid_im = -np.ones(render_shape)
    render_u = np.floor(u * render_shape[1] / img_shape[1]).astype(int)
    render_v = np.floor(v * render_shape[0] / img_shape[0]).astype(int)
    lid_im[render_v, render_u] = d

    if mask:
        return np.ma.array(lid_im, mask=lid_im == -1, fill_value=0)
    return lid_im


def quantify_img_error_lidar(predicted_img,
                             ground_truth_3d, Tr=HuskyCalib.T_cam0_vel0, K=HuskyCalib.left_camera_matrix,
                             gt_is_sparse_img=False, use_MSE=True,
                             generate_plots=False, normalise_by_test_pts=True):
    r"""
    Input a predicted depth image and Ground truth
        - Ground truth can be sparse image or lidar points

    This function will:
        Densify sparse Ground Truth
        Compare the two and give a quantitative output for error

        Choice of error metric: Sum of Squared Error (SSE), Mean Squared Error (MSE)

    Returns: Error
    """
    if np.all(np.greater(ground_truth_3d.shape, 3)):  # if input is an image or points
        if predicted_img.shape != ground_truth_3d.shape:
            ground_truth_3d = cv2.resize(ground_truth_3d, predicted_img.shape[::-1])

            if gt_is_sparse_img:
                dense_depth = make_sparse_dense(ground_truth_3d)
    else:
        dense_depth = make_sparse_dense(render_lidar(ground_truth_3d, Tr=Tr, K=K, render_shape=predicted_img.shape))

    e_im = (dense_depth - predicted_img) ** 2

    if use_MSE:
        e = e_im.sum() / e_im.count()
    else:
        e = np.sum(e_im)

    return e


def error_heatmap(predicted_img, ground_truth_3d, Tr=HuskyCalib.T_cam0_vel0, K=HuskyCalib.left_camera_matrix,
                  gt_is_sparse_img=False, normalise_by_mean=True, colourmap=cm.get_cmap('jet')):
    r"""
    Returns Heat map of error
    """

    if np.all(np.greater(ground_truth_3d.shape, 3)):  # if input is an image or points
        if predicted_img.shape != ground_truth_3d.shape:
            ground_truth_3d = cv2.resize(ground_truth_3d, predicted_img.shape[::-1])

            if gt_is_sparse_img:
                dense_depth = make_sparse_dense(ground_truth_3d)
    else:
        dense_depth = make_sparse_dense(render_lidar(ground_truth_3d, Tr=Tr, K=K, render_shape=predicted_img.shape))

    emap = ((predicted_img - dense_depth) ** 2)

    if normalise_by_mean:
        emap = emap / emap.mean()

    return colourmap(emap)

def get_pt_to_pt_error(sample_pts, gt_pts, use_MSE=True):
    e_im = (sample_pts - gt_pts) ** 2
    if use_MSE:
        e = e_im.sum() / e_im.count()
    else:
        e = np.sum(e_im)

    return e

def get_img_pt_to_pt_error(depth_im, uv, d, normalise_by_mean=False, img_shape=IMAGE_SHAPE, use_MSE=True):

    u, v = np.floor(uv).astype(int).T
    render_u = np.floor(u * depth_im.shape[1] / img_shape[1]).astype(int)
    render_v = np.floor(v * depth_im.shape[0] / img_shape[0]).astype(int)
    sampled_pts = depth_im[render_v, render_u]

    return get_pt_to_pt_error(sampled_pts, d, use_MSE=use_MSE)


def get_iterative_TR_error(imgl, imgr, lidar, use_MSE = True):
    mesh, pts = ITR.iterative_recon(img_l=imgl, img_r=imgr, num_iterations=RESAMPLING_ITERATIONS,
                                    num_pts_per_resample=25, eval_resampling_costs=False, frame=None,
                                    before_after_plots=False, plot_its=False, save_fig_path=None, verbose=False)

    uv = pts[:,:2]
    d = pts[:,2]
    e = get_img_pt_to_pt_error(render_lidar(lidar,img_shape=imgl.shape),uv=uv, d=d , use_MSE=use_MSE)
    return e


if __name__ == "__main__":
    import pandas as pd
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Data path locations:

    data_dir = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route C"
    training_dir = "/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train_Route_C"
    test_dir = "/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test_Route_C"
    # test_dir = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/Calibration and Lab/2022_04_29_09_29_47"
    test_filenames = os.path.join(test_dir, "test_file_list.txt")
    train_filenames = os.path.join(training_dir, 'training_file_list.txt')
    output_directory = '/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test_Route_C/predictions'
    checkpoint_dir = '/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train1/tmp/Husky5000/Husky'

    out_save_dir = os.path.join(output_directory, "output/image_00/data")
    query_save_dir = os.path.join(output_directory, "input/image_00/data")

    dataset = husky.Dataset_Handler(test_dir)
    predictions = os.listdir(out_save_dir)

    # --------------------------------------------------------------------------------------------------------

    jet = cm.get_cmap('jet')

    """
        - Load in predictions
        - Get matching frame number
        - Get corresponding lidar
        
        - Get error
        
        Nice to have next: 
        - Comparison of error with the tunable reconstruction
        - Comparison of time
    
    """

    frame_nums = []
    for q in os.listdir(query_save_dir):
        match = np.where(dataset.left_image_files == q)[0]
        if len(match):
            frame_nums.append(match[0])
        else:
            frame_nums.append(None)
    frame_nums = pd.Series(frame_nums)
    print(f"Matched queries for {np.sum(frame_nums != None)} of {len(frame_nums)} predictions")

    first_img = cv2.imread(os.path.join(dataset.left_image_path, dataset.left_image_files[0]))
    disp_img = cv2.imread(os.path.join(out_save_dir, predictions[0]))
    PRED_SHAPE = disp_img.shape

    MSE_errors = -np.ones(len(frame_nums))
    SSE_errors = -np.ones(len(frame_nums))
    img_errors = -np.ones(len(frame_nums))
    itr_errors = -np.ones(len(frame_nums))
    plotflag = True

    for i, frame in enumerate(frame_nums):
        """
         Get lidar gt
        Render Lidar

        Get Pydepth Prediction (load img for now. Later get eager execution)
        Get Iterative resampling
        Get aru core bindings estimate

        Eval error of PyDepth
        Eval error of Iterative Resampling
        Eval error of arucore stereo estimate (as stereo proxy)

        TODO Error eval on the fly when predicting
        TODO Make pretty graph animation
        TODO Iterative Resampling of PyDepth predictions


        """
        velo = dataset.get_lidar(frame)
        # img = cv2.cvtColor(dataset.get_cam0(frame), cv2.COLOR_BRG2RGB)
        img = dataset.get_cam0(frame)

        u, v, gt_d = lidar_to_img_frame(velo, HuskyCalib.T_cam0_vel0, dataset.left_camera_matrix, img_shape=img.shape)
        velo_cam = np.floor(np.stack((u, v, gt_d), axis=1)).astype(int)

        pred = cv2.imread(os.path.join(out_save_dir, predictions[i]), cv2.IMREAD_GRAYSCALE)
        pred = np.ma.array(pred, mask=(pred == 0), fill_value=1e-3, dtype=np.float32)

        img_errors[i] = quantify_img_error_lidar(pred, velo_cam)
        MSE_errors[i] = get_img_pt_to_pt_error(pred, velo_cam[:,:2], d=gt_d)
        SSE_errors[i] = get_img_pt_to_pt_error(pred, velo_cam[:,:2], d=gt_d)

        itr_errors[i] = get_iterative_TR_error(dataset.get_cam0(frame), dataset.get_cam1(frame), velo_cam[::20])

        if plotflag:

            print("Generating Plots")
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            fig.tight_layout(pad=2)
            for row in ax:
                for a in row:
                    a.axis('off')

            ax[1, 0].imshow(dataset.get_cam0(frame))
            im = ax[1, 0].scatter(u, v, c=gt_d, cmap='jet', s=3)
            divider = make_axes_locatable(ax[1, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax[1, 0].set_title("Lidar Image")

            im = ax[1, 1].imshow(pred, 'jet')
            divider = make_axes_locatable(ax[1, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax[1, 1].set_title("Predicted output")

            im = ax[0, 1].imshow(error_heatmap(pred, velo_cam[::20]))
            divider = make_axes_locatable(ax[0, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax[0, 1].set_title("Error Map")

            ax[0, 0].imshow(dataset.get_cam0(frame))
            ax[0, 0].set_title("Query img")
            fig.suptitle(f"Output and Error for frame {frame}")
            plt.show()
            plotflag = False

    plt.plot(frame_nums, img_errors)
    plt.title("Mean Squared Errors over frames")
    plt.show()

    plt.plot(frame_nums, MSE_errors)
    plt.title("Mean Squared Errors sampled at lidar over frames")
    plt.show()

    plt.plot(frame_nums, itr_errors)
    plt.title("Mean Squared Errors for Iterative Reconstruction")
    plt.show()

    print(f"Summary of the errors over the {len(frame_nums)} frames:\n"
          f"-------------------------------------------------------------\n"
          f"\tProjected Lidar Pt-to-Pt Errors:\n"
          f"\t\t- PyDepth Mean MSE {np.mean(MSE_errors):.2f}\n"
          f"\t\t- PyDepth Mean SSE {np.mean(SSE_errors):.2f}\n"
          f"\n\tInterpolated Lidar Errors:\n"
          f"\t\t- PyDepth Mean MSE {np.mean(img_errors):.2f}\n"
          f"\t\t- Iterative Reconstruction @(3 resamle iterations and 25pts) Mean MSE:{np.mean(itr_errors):.2f}\n"
          f"")

