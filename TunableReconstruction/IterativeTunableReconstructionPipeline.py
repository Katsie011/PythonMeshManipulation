r"""
Implementation of High Resolution Tunable Reconstruction from https://arxiv.org/pdf/1511.00758.pdf
Implemented by:                                                         Michael Katsoulis March 2022


This is the pipeline implementation after experimenting with the concept in a jupyter notebook.

"""

# -----------------------------------------------------------------------------------------------------------------
#       Importing libraries
# -----------------------------------------------------------------------------------------------------------------

import sys

if "aru_core_lib" not in sys.modules:
    # you may need a symbolic link to the build of aru_core library
    import reconstruction.aru_core_lib.aru_py_mesh as aru_py_mesh

import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import cv2

# You may need to use symlinks to pull the corresponding repos into the correct places.
import reconstruction.TunableReconstruction.Functions_TunableReconstruction as TR_func

# -----------------------------------------------------------------------------------------------------------------
#       Declaring Hyperparameters
# -----------------------------------------------------------------------------------------------------------------
from reconstruction.HyperParameters import *


# -----------------------------------------------------------------------------------------------------------------
#       Extracting Features
# -----------------------------------------------------------------------------------------------------------------

def extract_fts(im_left, im_right, config_file="mesh_depth.yaml", validate_pts=True, verbose=True):
    r"""
    This just extracts features using an ORB feature extractor.
    The number of features is set in the config file
    """
    if verbose: print("Extracting features")
    if verbose: print("\tHave images of shape ", im_left.shape)
    if verbose: print("\t Expanding dimensions to ", np.expand_dims(im_left, -1).shape)

    # if verbose: print("\tMaking Depth estimator")
    # depth_est = aru_py_mesh.PyDepth(config_file) # this is no longer necessary if you import the HyperParameters file

    if verbose: print("\tExtracting depth")

    if verbose: print("\tGetting sparse depth")
    if len(im_left.shape) == 2:
        im_left = cv2.cvtColor(im_left, cv2.COLOR_GRAY2RGB)
        im_right = cv2.cvtColor(im_right, cv2.COLOR_GRAY2RGB)

    sparse_depth_img = depth_est.create_sparse_depth(im_left, im_right)

    sparse_pts = np.where(sparse_depth_img > 0)
    ft_uvd = np.array((sparse_pts[1], sparse_pts[0], sparse_depth_img[sparse_pts])).T

    if validate_pts: ft_uvd = ft_uvd[ft_uvd[:, 2] < MAX_DISTANCE]  # Validating points
    if verbose: print(f"\tGot {len(sparse_pts[0])} depth points back. \n {len(sparse_pts[0])} points were valid")
    if verbose: print(f"\t{len(np.where(sparse_depth_img < 0)[0])} recieved points were < 0")
    return ft_uvd


# -----------------------------------------------------------------------------------------------------------------
#       Cost Calculation
# -----------------------------------------------------------------------------------------------------------------


def calculate_costs(img_l, img_r, interpolated, ft_uvd, K=K, t=t, verbose=True):
    r"""
    Get the census cost for the interpolated points.
    Census compared for the disparity at pts in left and right img
    """

    if verbose: print("Calculating costs")
    new_census = np.abs(TR_func.get_disparity_census_cost(interpolated, img_l, img_r, K, t, num_disparity_levels=5))

    # max and min costs are set using distribution curves
    MAX_COST = new_census[:, new_census.shape[1] // 2].mean() \
               - new_census[:, new_census.shape[1] // 2].std()  # Using shifting boundaries
    MIN_COST = new_census.min(axis=1).mean() - new_census.min(axis=1).std()  # Statistically defined minimum
    FT_COSTS = np.abs(TR_func.get_disparity_census_cost(ft_uvd, img_l, img_l, K, t, num_disparity_levels=1)).mean()
    if verbose: print("\t-----------------------------")
    if verbose: print("\tCost Calculation:")

    if verbose: print(f"\tMean {new_census.mean()}")
    if verbose: print(f"\tMax {new_census.max()}")
    if verbose: print(f"\tMin {new_census.min()}")

    if verbose: print(f"\tSetting MAX_COST to {MAX_COST}")
    if verbose: print(f"\tSetting MIN_COST to {MIN_COST}")
    if verbose: print(f"\tAverage cost from features for mesh construction was {FT_COSTS}")

    min_cost_idx = new_census.min(axis=1) < MIN_COST

    c_g = -1 * np.ones((len(new_census), 4))  # (u,v,d,c)
    c_b = c_g.copy()

    # good points are those with the lowest cost
    c_g = np.hstack((interpolated[min_cost_idx, :2],
                     TR_func.depth_to_disparity(interpolated[min_cost_idx, 2], K, t).reshape((-1, 1)),
                     new_census.min(axis=1)[min_cost_idx].reshape((-1, 1))))
    max_cost_idx = ~min_cost_idx
    c_b = np.hstack((interpolated[max_cost_idx, :2],
                     TR_func.depth_to_disparity(interpolated[max_cost_idx, 2], K, t).reshape((-1, 1)),
                     new_census.min(axis=1)[max_cost_idx].reshape((-1, 1))))

    if verbose: print(f"\tTotal num census cost pts {len(new_census)}")
    if verbose: print("\t\tNumber of good interpolated points:", np.sum(np.sum(c_g, axis=1) != -c_g.shape[1]))
    if verbose: print("\t\tNumber of bad interpolated points:", np.sum(np.sum(c_b, axis=1) != -c_b.shape[1]))

    return new_census, c_g, c_b


# -----------------------------------------------------------------------------------------------------------------
#       Resampling Iteratively
# -----------------------------------------------------------------------------------------------------------------
def resample_iterate(imgl, imgr, pts_resampling, eval_resampling_costs=False, verbose=True):
    r"""
    This samples more points around those that have high cost

    These points then have their depth resampled by stereo matching using a census cost
    Resampled points are returned.
    """
    if verbose: print("Support Resampling")

    support_pts = TR_func.support_resampling(imgl, imgr, pts_resampling)
    support_pts = support_pts[np.logical_and(np.logical_and(
        np.all(support_pts > 0, axis=1), (support_pts[:, 0] < imgl.shape[1])), (support_pts[:, 1] < imgl.shape[0]))]

    # -----------------------------------------------------------------------------------------------------------------
    #       Adjusting depth for resampled points
    # -----------------------------------------------------------------------------------------------------------------
    # Searching for points along epipolar lines:
    new_d = TR_func.get_depth_with_epipolar(imgl, imgr, support_pts, K=K, t=t, F=None, use_epipolar_line=False)
    resampled = np.stack((support_pts[:, 0], support_pts[:, 1], new_d), axis=-1)
    resampled = resampled[resampled[:, 2] <= MAX_DISTANCE]

    if eval_resampling_costs:
        c_bf = TR_func.get_disparity_census_cost(support_pts, imgl, imgr, K, t, num_disparity_levels=1).mean()
        c_af = TR_func.get_disparity_census_cost(resampled, imgl, imgr, K, t, num_disparity_levels=1).mean()
        if verbose: print(f"\tResampled {len(support_pts)} support points")
        if verbose: print(f"\tAvg resampling cost before: {c_bf} \t after: {c_af}")

    return resampled


def stereo_iterative_recon(img_l, img_r, num_iterations=RESAMPLING_ITERATIONS, num_pts_per_resample=25,
                           eval_resampling_costs=False, frame=None, before_after_plots=False,
                           plot_its=False, save_fig_path=None, verbose=False):
    r"""
    Combines all the stereo iterative steps into one function

    Returns:
        depth_mesh, 3D depth_pts
    """
    # --------------------------------------
    #       Getting feature depth
    # --------------------------------------
    ft_uvd = extract_fts(img_l, img_r)
    # Building mesh
    depth_mesh = Delaunay(ft_uvd[:, :2])
    depth_mesh_pts = ft_uvd.copy()

    if before_after_plots:
        print("Plotting initial points")
        fig_bna, ax_bna = plt.subplots(2, 2, figsize=(22, 7.5))
        fig_bna.tight_layout(pad=3)
        ax_bna[0, 0].imshow(img_l, 'gray')
        sc = ax_bna[0, 0].scatter(depth_mesh_pts[:, 0], depth_mesh_pts[:, 1], c=depth_mesh_pts[:, 2], cmap='jet')
        divider = make_axes_locatable(ax_bna[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label("Depth [m]")
        ax_bna[0, 0].set_title("Initial Points")

        ax_bna[0, 1].imshow(img_l, 'gray')
        # plot_mesh(mesh, mesh_pts, a=ax[0, 1])
        ax_bna[0, 1].triplot(depth_mesh_pts[:, 0], depth_mesh_pts[:, 1], depth_mesh.simplices.copy())
        ax_bna[0, 1].set_title("Initial mesh")
        ax_bna[0, 0].axis('off')
        ax_bna[0, 1].axis('off')
        # plt.show()
        print("Done plotting")

    """ 
    Now to iterate over the resampling part:
         Interpolate mesh
         Get cost for interpolated points
         Resample worst n points
         Add good points and resampled points to mesh vertices list
         Repeat
    """
    # --------------------------------------
    #       Interpolating
    # --------------------------------------
    still_to_resample = TR_func.interpolate_pts(img_l, depth_mesh, depth_mesh_pts, verbose=verbose)

    if plot_its:
        fig_its, ax_its = plt.subplots(num_iterations, 2, figsize=(22, 3.7 * num_iterations))
    for its in range(num_iterations):

        # --------------------------------------
        #       Cost Calculation
        # --------------------------------------

        c_interpolated, good_c_pts, bad_c_pts = calculate_costs(img_l, img_r, still_to_resample,
                                                                depth_mesh_pts, verbose=verbose)
        idx = np.argsort(bad_c_pts[:, -1])[::-1]
        resampling_pts = bad_c_pts[idx[:num_pts_per_resample], :3]
        still_to_resample = bad_c_pts[idx[num_pts_per_resample:]]
        # cost_bad_pts = n x [u, v, d, c]
        resampled_pts = resample_iterate(img_l, img_r, resampling_pts, eval_resampling_costs=eval_resampling_costs,
                                         verbose=verbose)

        # --------------------------------------
        #       Merging with mesh
        # --------------------------------------

        depth_mesh_pts = np.vstack((depth_mesh_pts, good_c_pts[:, :3], resampled_pts))
        depth_mesh_pts = depth_mesh_pts[depth_mesh_pts[:, 2] < MAX_DISTANCE]
        depth_mesh = Delaunay(depth_mesh_pts[:, :2])

        # --------------------------------------
        #       Plots
        # --------------------------------------
        if plot_its:
            fig_its.tight_layout()
            ax_its[its, 0].imshow(img_l, 'gray')
            ax_its[its, 0].axis('off')
            sc = ax_its[its, 0].scatter(depth_mesh_pts[:, 0], depth_mesh_pts[:, 1], c=depth_mesh_pts[:, 2], cmap='jet')
            divider = make_axes_locatable(ax_its[its, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label("Depth [m]")

            ax_its[its, 1].imshow(img_l, 'gray')
            ax_its[its, 1].axis('off')
            # ax_its[1].triplot(mesh_pts[:,0], mesh_pts[:,1], mesh.simplices.copy())
            sc = ax_its[its, 1].scatter(resampled_pts[:, 0], resampled_pts[:, 1], c=resampled_pts[:, 2], cmap='jet')
            divider = make_axes_locatable(ax_its[its, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label("Depth [m]")
            ax_its[its, 1].set_title(f"Resampling Iteration {its}")

    if plot_its:
        if frame is not None:
            title_its = f"Plot of iterations for frame: #{frame}"
        else:
            title_its = "Plot of iterations"
        fig_its.suptitle(title_its, fontsize=24)
        if save_fig_path:
            save_fig = os.path.join(save_fig_path, title_its + ".png")
            fig_its.savefig(save_fig)
        plt.show()

    if before_after_plots:
        ax_bna[1, 0].imshow(img_l, 'gray')
        sc = ax_bna[1, 0].scatter(depth_mesh_pts[:, 0], depth_mesh_pts[:, 1], c=depth_mesh_pts[:, 2], cmap='jet')
        divider = make_axes_locatable(ax_bna[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label("Depth [m]")
        ax_bna[1, 0].set_title("Final Mesh Points")

        ax_bna[1, 1].imshow(img_l, 'gray')
        # plot_mesh(mesh, mesh_pts, a=ax[1, 1])
        ax_bna[1, 1].triplot(depth_mesh_pts[:, 0], depth_mesh_pts[:, 1], depth_mesh.simplices.copy())
        ax_bna[1, 1].set_title("Final Mesh")

        if frame is not None:
            title = f"Tunable Reconstruction For Frame: #{frame}"
        else:
            title = "Tunable Reconstruction"

        fig_bna.suptitle(title, fontsize=24)
        ax_bna[1, 0].axis('off')
        ax_bna[1, 1].axis('off')

        if save_fig_path is not None:
            path = os.path.join(save_fig_path, title + ".png")
            fig_bna.savefig(path)
        plt.show()

    return depth_mesh, depth_mesh_pts


def depth_img_iterative_recon(img_l, img_r, depth_prediction, detector='sift', return_plot=False, frame_num=0,
                              img_shape=IMAGE_SHAPE, verbose=False):


    height, width = img_shape[:2]
    mesh_pts = TR_func.get_depth_pts(detector, img_l, depth_prediction)
    depth_mesh = Delaunay(mesh_pts[:, :2])
    u, v, d = mesh_pts.T

    # ----------------------------------------------------
    #     Interpolating
    # ----------------------------------------------------

    to_resample = TR_func.interpolate_pts(img_l, depth_mesh, mesh_pts, verbose=verbose)

    # ----------------------------------------------------
    #       Cost Calculation
    # ----------------------------------------------------

    c_interpolated, good_c_pts, bad_c_pts = TR_func.calculate_costs(img_l, img_r, to_resample,
                                                                    mesh_pts, verbose=verbose)

    idx = np.argsort(bad_c_pts[:, -1])[::-1]
    num_pts_per_resample = 25
    eval_resampling_costs = False
    resampling_pts = bad_c_pts[idx[:num_pts_per_resample], :3]
    # still_to_resample = bad_c_pts[idx[num_pts_per_resample:]]
    # cost_bad_pts of shape{n x [u, v, d, c]}
    resampled_pts = TR_func.resample_iterate(img_l, img_r, resampling_pts,
                                             eval_resampling_costs=eval_resampling_costs,
                                             verbose=verbose)

    new_mesh_pts = np.vstack((mesh_pts, good_c_pts[:, :3], resampled_pts))
    new_mesh_pts = new_mesh_pts[new_mesh_pts[:, 2] < MAX_DISTANCE]
    new_mesh = Delaunay(new_mesh_pts[:, :2]) # this can always be calculated quickly

    if not return_plot:
        return new_mesh_pts

    else:
        dpi = 40
        # figsize = 2*height / float(dpi), 2*width / float(dpi)
        figsize = width / float(dpi), height / float(dpi)
        fig, ax = plt.subplots(2, 3, figsize=figsize)  # , dpi=dpi)
        fig.tight_layout(pad=2)

        disp_im = ax[0, 0].imshow(disp[0, :, :, 0].squeeze(), 'jet')
        ax[0, 0].set_title(f"Disparity img frame {frame_num}")
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(disp_im, cax=cax, orientation='vertical')

        depth_im = ax[0, 1].imshow(depth_prediction, 'jet')
        ax[0, 1].set_title(f"Bounded depth img frame {frame_num}")
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(depth_im, cax=cax, orientation='vertical')

        ax[0, 2].hist(depth_prediction.flatten(), bins=100)
        ax[0, 2].set_title("Histogram of depth predictions")

        ax[1, 0].imshow(img_l)
        sc = ax[1, 0].scatter(u, v, c=d, s=10, cmap='jet')
        ax[1, 0].set_title(f"Scattered pts before Tunable Recon - Frame {frame_num}")
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sc, cax=cax, orientation='vertical')

        ax[1, 1].imshow(img_l)
        u2, v2, d2 = new_mesh_pts.T
        sc1 = ax[1, 1].scatter(u2, v2, c=d2, s=10, cmap='jet')
        ax[1, 1].set_title(f"Scattered pts after Tunable Recon - Frame {frame_num}")
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

        return new_mesh_pts, plot


if __name__ == "__main__":
    import pykitti

    print("Loading database")
    base = r"/home/kats/Datasets/KITTI_cvlibs/"
    date = "2011_09_26"
    drive = "0001"
    data = pykitti.raw(base, date, drive)

    plot_iterations = False

    frames = np.arange(0, len(data.cam0_files), 20)
    for frame in frames:
        img_left = np.array(data.get_cam0(frame), dtype=np.uint8)
        img_right = np.array(data.get_cam1(frame), dtype=np.uint8)

        mesh, pts = stereo_iterative_recon(img_left, img_right,
                                           save_fig_path="/PythonMeshManipulation/Figures/TunableReconstruction",
                                           before_after_plots=True)

        # lidar_img = y = P(i) @ R(0) @ Tvelo_cam @ x
