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
import numpy as np
import cv2

from reconstruction.HyperParameters import *
import reconstruction.TunableReconstruction.Functions_TunableReconstruction as TR_func
import utilities.HuskyDataHandler as husky


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
#       Globals
# ----------------------------------------------------------------------------



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

def sample_img(img, pts_2d):
    r"""
    Returns the values in the image for those points in:
        u, v, sample
    """
    if pts_2d.shape[1]==2:
        pts_2d = pts_2d.T

    u,v = np.floor(pts_2d).astype(int)

    samples = img[v,u]

    return u,v,samples


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
                 render_shape=None, mask=True):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
       numpy array. Returns the 2D projection of the points that
       are in front of the camera only an the corresponding 3D points.'''

    if len(img_shape) == 3:
        img_shape = img_shape[:2]

    if render_shape is not None:
        if len(render_shape) == 3:
            render_shape = render_shape[:2]
    else:
        render_shape = img_shape

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
                             ground_truth_3d, Tr=HuskyCalib.T_cam0_vel0, K=HuskyCalib.left_camera_matrix, baseline=None,
                             convert_to_disp=False, gt_is_sparse_img=False, use_MSE=True,
                             generate_plots=False, normalise_by_test_pts=True, ):
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

    if convert_to_disp:
        predicted_img = TR_func.disparity_to_depth(predicted_img, K=K, t=baseline)
        dense_depth = TR_func.disparity_to_depth(dense_depth, K=K, t=baseline)

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


def pt_to_pt_emap(predicted_img, gt_cam_pts, Tr=HuskyCalib.T_cam0_vel0, K=HuskyCalib.left_camera_matrix,
                  gt_is_sparse_img=False, normalise_by_mean=True, mask=False, colourmap=cm.get_cmap('jet'),
                  img_shape=IMAGE_SHAPE, dilation=10):
    r"""
    Returns Heat map of error but only at the sample pts
    """

    idx, idy = np.floor(gt_cam_pts.T[:2]).astype(int)
    if (predicted_img.shape[0] < img_shape[0]) or (predicted_img.shape[1] < img_shape[1]):
        predicted_img = cv2.resize(predicted_img, img_shape[:2][::-1])

    if colourmap is None:
        emap = np.zeros((predicted_img.shape[0], predicted_img.shape[1]))
    else:
        emap = np.zeros((predicted_img.shape[0], predicted_img.shape[1], 4))
    es = ((predicted_img[idy, idx] - gt_cam_pts[:, 2]) ** 2)

    if normalise_by_mean:
        es = es / es.mean()

    if colourmap is None:
        emap[idy, idx] = es
    else:
        emap[idy, idx] = colourmap(es)

    emap = cv2.dilate(emap, np.ones((dilation, dilation)))

    if mask:
        emap = np.ma.masked_where(emap == 0, emap)
        return emap

    return emap


def get_pt_to_pt_error(sample_pts, gt_pts, use_MSE=True):
    e_im = (sample_pts - gt_pts) ** 2
    if use_MSE:
        e = e_im.sum() / e_im.size
    else:
        e = np.sum(e_im)

    return e


def get_img_pt_to_pt_error(depth_im, uv, d, normalise_by_mean=False, img_shape=IMAGE_SHAPE, use_MSE=True):
    u, v = np.floor(uv).astype(int).T
    render_u = np.floor(u * depth_im.shape[1] / img_shape[1]).astype(int)
    render_v = np.floor(v * depth_im.shape[0] / img_shape[0]).astype(int)
    sampled_pts = depth_im[render_v, render_u]

    return get_pt_to_pt_error(sampled_pts, d, use_MSE=use_MSE)


def get_iterative_TR_error(imgl, imgr, lidar, use_MSE=True):
    mesh, pts = ITR.stereo_iterative_recon(img_l=imgl, img_r=imgr, num_iterations=RESAMPLING_ITERATIONS,
                                           num_pts_per_resample=25, eval_resampling_costs=False, frame=None,
                                           before_after_plots=False, plot_its=False, save_fig_path=None, verbose=False)

    uv = pts[:, :2]
    d = pts[:, 2]
    e = get_img_pt_to_pt_error(render_lidar(lidar, img_shape=imgl.shape), uv=uv, d=d, use_MSE=use_MSE)
    return e

def rough_lidar_render(velo, k_dilate = 5, k_blur = 20, img_shape = IMAGE_SHAPE, mask=True):
    lid_im = np.zeros(img_shape[:2])
    u,v,disp = velo.T
    u,v = np.floor((u,v)).astype(int)
    lid_im[v, u] = disp

    lid_im = cv2.blur(cv2.dilate(lid_im, np.ones((k_dilate, k_dilate))), (k_blur, k_blur))
    if mask:
        lid_im = np.ma.masked_where(lid_im==0, lid_im)
    return lid_im


if __name__ == "__main__":
    import pandas as pd
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    import reconstruction.TunableReconstruction.IterativeTunableReconstructionPipeline as ITR

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

        velo = dataset.get_lidar(i)
        img = dataset.get_cam0(i)

        u, v, gt_depth = lidar_to_img_frame(velo, HuskyCalib.T_cam0_vel0, dataset.left_camera_matrix,
                                            img_shape=img.shape)
        gt_disparity = TR_func.depth_to_disparity(gt_depth, dataset.left_camera_matrix, dataset.baseline)
        velo_cam = np.floor(np.stack((u, v, gt_disparity), axis=1)).astype(int)

        pred = cv2.imread(os.path.join(out_save_dir, predictions[i]), cv2.IMREAD_GRAYSCALE)
        pred = TR_func.depth_to_disparity(pred, dataset.left_camera_matrix, dataset.baseline)
        pred = np.ma.array(pred, mask=(pred == 0), fill_value=1e-3, dtype=np.float32)

        print(pred.max())
        # Are the errors actually correct?
        # What is the range in disparities?

        img_errors[i] = quantify_img_error_lidar(pred, velo_cam)
        MSE_errors[i] = get_img_pt_to_pt_error(pred, velo_cam[:, :2], d=gt_disparity)
        SSE_errors[i] = get_img_pt_to_pt_error(pred, velo_cam[:, :2], d=gt_disparity)

        # itr_errors[i] = get_iterative_TR_error(dataset.get_cam0(frame), dataset.get_cam1(frame), velo_cam[::20])

        if plotflag:

            print("Generating Plots")
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            fig.tight_layout(pad=3)
            for row in ax:
                for a in row:
                    a.axis('off')

            ax[1, 0].imshow(dataset.get_cam0(frame))
            im = ax[1, 0].scatter(u, v, c=gt_disparity, cmap='jet', s=3)
            divider = make_axes_locatable(ax[1, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label("Disparity")
            ax[1, 0].set_title("Lidar Image in disparity")

            im = ax[0, 1].imshow(pred, 'jet')
            divider = make_axes_locatable(ax[0, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label("Disparity")
            ax[0, 1].set_title("Predicted output")

            # im = ax[1, 1].imshow(error_heatmap(pred, velo_cam[::20]))
            im = ax[1, 1].imshow(pt_to_pt_emap(pred, velo_cam, mask=True, colourmap=None), 'jet')
            divider = make_axes_locatable(ax[1, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label("Disparity")
            ax[1, 1].set_title(f"Error Map. MSE:{MSE_errors[i]:.2f}")

            ax[0, 0].imshow(dataset.get_cam0(frame))
            ax[0, 0].set_title("Query img")
            fig.suptitle(f"Output and Error for frame {frame}")

            # plt.show()
            # plotflag = False

            fig.canvas.draw()
            canvas = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                   sep='')
            canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imshow('output', canvas)
            cv2.waitKey(40)

    plt.show()

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
