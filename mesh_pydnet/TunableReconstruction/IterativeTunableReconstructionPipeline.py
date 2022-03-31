r"""
Implementation of High Resolution Tunable Reconstruction from https://arxiv.org/pdf/1511.00758.pdf
Implemented by:                                                         Michael Katsoulis March 2022


This is the pipeline implementation after experimenting with the concept in a jupyter notebook.

"""

# -----------------------------------------------------------------------------------------------------------------
#       Importing libraries
# -----------------------------------------------------------------------------------------------------------------
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
import sys

sys.path.insert(0, "/ModularFiles")
# import DatasetHandler
# import ImgFeatureExtactorModule as ft
# import TriangleUtilityFunctions
# import Kitti_Dataset_Files_Handler
# import pykitti
from HyperParameters import *

# /home/kats/Documents/My Documents/Datasets/KITTI_cvlibs/2011_09_26/2011_09_26_drive_0001_sync


from Functions_TunableReconstruction import *

# fig, ax = plt.subplots(1,2)
# ax[0].imshow(np.array(data.get_cam0(0)))
# ax[1].imshow(np.array(data.get_cam1(0)))
#
# ax[0].set_title("Test Left Img")
# ax[1].set_title("Test Right Img")
# plt.show()

# -----------------------------------------------------------------------------------------------------------------
#       Declaring Hyperparameters
# -----------------------------------------------------------------------------------------------------------------
sys.path.insert(0, '/home/kats/Documents/Repos/aru-core/build/lib/')
import aru_py_mesh


# -----------------------------------------------------------------------------------------------------------------
#       Extracting Features
# -----------------------------------------------------------------------------------------------------------------

def extract_fts(data, imgl, imgr, validate_pts=True, verbose=True):
    print("Extracting features")
    if verbose: print("Have images of shape ", imgl.shape)
    if verbose: print("\t Expanding dimensions to ", np.expand_dims(imgl, -1).shape)

    if verbose: print("Making Depth estimator")
    depth_est = aru_py_mesh.PyDepth("mesh_depth.yaml")

    if verbose: print("Extracting depth")

    if verbose: print("Getting sparse depth")
    img_left = cv2.cvtColor(imgl, cv2.COLOR_GRAY2RGB)
    img_right = cv2.cvtColor(imgr, cv2.COLOR_GRAY2RGB)
    sparse_depth_img = depth_est.create_sparse_depth(img_left, img_right)
    # sparse_depth_img = np.abs(sparse_depth_img)
    # plt.imshow(cv2.dilate((sparse_depth_img>0).astype(np.uint8), np.ones((4,4))), 'gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    sparse_pts = np.where(sparse_depth_img > 0)
    ft_uvd = np.array((sparse_pts[1], sparse_pts[0], sparse_depth_img[sparse_pts])).T

    if validate_pts: ft_uvd = ft_uvd[ft_uvd[:, 2] < MAX_DISTANCE]  # Validating points
    if verbose: print(f"Got {len(sparse_pts[0])} depth points back. \n {len(sparse_pts[0])} points were valid")
    if verbose: print(f"{len(np.where(sparse_depth_img < 0)[0])} recieved points were < 0")
    return ft_uvd


def interpolate_pts(imgl, d_mesh, ft_uvd, verbose=False):
    δ = cv2.Laplacian(imgl, cv2.CV_64F)
    δ = np.abs(δ)

    idx = np.argpartition(δ.flatten(), -INTERPOLATING_POINTS)[-INTERPOLATING_POINTS:]
    gradient_pts = np.unravel_index(idx, imgl.shape)
    interpolated_uv = np.stack((gradient_pts[1], gradient_pts[0]), axis=-1)
    interpolated_pts = barycentric_interpolation(d_mesh, ft_uvd, interpolated_uv)
    if verbose: print(f"Interpolated and returning {len(interpolated_pts)} points")
    return interpolated_pts


# -----------------------------------------------------------------------------------------------------------------
#       Cost Calculation
# -----------------------------------------------------------------------------------------------------------------


def calculate_costs(imgl, imgr, interpolated, ft_uvd, verbose=True):
    print("Calculating costs")
    new_census = np.abs(get_disparity_census_cost(interpolated, imgl, imgr, K, t, num_disparity_levels=5))

    # MAX_COST = 0.8  # Hardcoded values
    # MIN_COST = 0.2  # Hardcoded values

    MAX_COST = new_census[:, new_census.shape[1] // 2].mean() \
               - new_census[:, new_census.shape[1] // 2].std()  # Using shifting boundaries
    MIN_COST = new_census.min(axis=1).mean() - new_census.min(axis=1).std()  # Statistically defined minimum
    FT_COSTS = np.abs(get_disparity_census_cost(ft_uvd, imgl, imgl, K, t, num_disparity_levels=1)).mean()
    if verbose: print("-----------------------------")
    if verbose: print("Cost Calculation:")

    if verbose: print(f"Mean {new_census.mean()}")
    if verbose: print(f"Max {new_census.max()}")
    if verbose: print(f"Min {new_census.min()}")

    if verbose: print(f"Setting MAX_COST to {MAX_COST}")
    if verbose: print(f"Setting MIN_COST to {MIN_COST}")
    if verbose: print(f"Average cost from features for mesh construction was {FT_COSTS}")

    min_cost_idx = new_census.min(axis=1) < MIN_COST

    # Using the best match in the window at this point. It might not be right
    # max_cost_idx = new_census.min(axis=1) > MAX_COST

    # c_g = np.nan*np.ones((len(new_census), 4)) # (u,v,d,c)
    c_g = -1 * np.ones((len(new_census), 4))  # (u,v,d,c)
    c_b = c_g.copy()

    # good points are those with the lowest cost
    c_g = np.hstack((interpolated[min_cost_idx, :2],
                     depth_to_disparity(interpolated[min_cost_idx, 2], K, t).reshape((-1, 1)),
                     new_census.min(axis=1)[min_cost_idx].reshape((-1, 1))))
    # c_g_d_arg = np.argmin(new_census, axis=1)
    # c_b_d_arg = np.argmax(new_census, axis=1)

    # # Using the best match (new_census.min)
    # c_b[max_cost_idx] = np.hstack((interpolated_pts[max_cost_idx, :2],
    #                                depth_to_disparity(interpolated_pts[max_cost_idx, 2], K, t).reshape((-1, 1)),
    #                                new_census.min(axis=1)[max_cost_idx].reshape((-1, 1))))

    # Choosing bad points as compliment of good points
    max_cost_idx = ~min_cost_idx
    c_b = np.hstack((interpolated[max_cost_idx, :2],
                     depth_to_disparity(interpolated[max_cost_idx, 2], K, t).reshape((-1, 1)),
                     new_census.min(axis=1)[max_cost_idx].reshape((-1, 1))))

    if verbose: print(f"Total num census cost pts {len(new_census)}")
    if verbose: print("Number of good interpolated points:", np.sum(np.sum(c_g, axis=1) != -c_g.shape[1]))
    if verbose: print("Number of bad interpolated points:", np.sum(np.sum(c_b, axis=1) != -c_b.shape[1]))

    return new_census, c_g, c_b


# -----------------------------------------------------------------------------------------------------------------
#       Resampling Iteratively
# -----------------------------------------------------------------------------------------------------------------
def resample_iterate(imgl, imgr, pts_resampling, eval_resampling_costs=False, verbose=True):
    print("Support Resampling")
    # pts_resampling = pts_to_still_resample[-num_resample:]
    # pts_to_still_resample = pts_to_still_resample[:-num_resample]

    support_pts = support_resampling(imgl, imgr, pts_resampling)
    support_pts = support_pts[np.logical_and(np.logical_and(
        np.all(support_pts > 0, axis=1), (support_pts[:, 0] < imgl.shape[1])), (support_pts[:, 1] < imgl.shape[0]))]

    # -----------------------------------------------------------------------------------------------------------------
    #       Adjusting depth for resampled points
    # -----------------------------------------------------------------------------------------------------------------
    # Searching for points along epipolar lines:
    new_d = get_depth_with_epipolar(imgl, imgr, support_pts, K=K, t=t, F=None, use_epipolar_line=False)
    resampled = np.stack((support_pts[:, 0], support_pts[:, 1], new_d), axis=-1)
    resampled = resampled[resampled[:, 2] <= MAX_DISTANCE]

    if eval_resampling_costs:
        c_bf = get_disparity_census_cost(support_pts, imgl, imgr, K, t, num_disparity_levels=1).mean()
        c_af = get_disparity_census_cost(resampled, imgl, imgr, K, t, num_disparity_levels=1).mean()
        if verbose: print(f"Resampled {len(support_pts)} support points")
        if verbose: print(f"Avg resampling cost before: {c_bf} \t after: {c_af}")

    return resampled
    # new_pts = np.vstack((new_pts, resampled))
    # ax[r, 0].imshow(imgl, 'gray')
    # ax[r, 0].axis('off')
    # sc = ax[r, 0].scatter(resampled[:, 0], resampled[:, 1], c=resampled[:, 2], cmap='jet')
    # divider = make_axes_locatable(ax[r, 0])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(sc, cax=cax)
    # cbar.set_label("Depth [m]")
    # ax[r, 0].set_title(f"Resampling iteration {r}", fontsize=18)
    #
    # ax[r, 1].imshow(img_left)
    # ax[r, 1].axis('off')
    # plot_mesh(Delaunay(resampled[:, :2]), resampled, a=ax[r, 1])
    # ax[r, 1].set_title("Resulting mesh", fontsize=18)


# ax[r + 1, 0].imshow(imgl, 'gray')
# ax[r + 1, 0].axis('off')
# sc = ax[r + 1, 0].scatter(new_pts[:, 0], new_pts[:, 1], c=new_pts[:, 2], cmap='jet')
# divider = make_axes_locatable(ax[r + 1, 0])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(sc, cax=cax)
# cbar.set_label("Depth [m]")
# ax[r + 1, 0].set_title(f"Final points after resampling", fontsize=18)
#
# ax[r + 1, 1].imshow(imgl, 'gray')
# ax[r + 1, 1].axis('off')
# plot_mesh(Delaunay(new_pts[:, :2]), new_pts, a=ax[r + 1, 1])
# ax[r, 1].set_title("Resulting mesh", fontsize=18)
#
# plt.show()


def iterative_recon(img_l, img_r, num_iterations=RESAMPLING_ITERATIONS, num_pts_per_resample=25,
                    eval_resampling_costs=False, frame = None, before_after_plots=False,
                    plot_its=False, save_fig_path=None, verbose=False):
    # --------------------------------------
    #       Getting feature depth
    # --------------------------------------
    ft_uvd = extract_fts(data, img_l, img_r)
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
    still_to_resample = interpolate_pts(img_l, depth_mesh, depth_mesh_pts, verbose=verbose)

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


if __name__ == "__main__":
    import ErrorEvaluation

    print("Loading database")
    base = r"/home/kats/Documents/My Documents/Datasets/KITTI_cvlibs/"
    date = "2011_09_26"
    drive = "0001"
    data = pykitti.raw(base, date, drive)

    plot_iterations = False

    frames = np.arange(0, len(data.cam0_files), 20)
    for frame in frames:
        img_left = np.array(data.get_cam0(frame), dtype=np.uint8)
        img_right = np.array(data.get_cam1(frame), dtype=np.uint8)

        mesh, pts = iterative_recon(img_left, img_right,
                                    save_fig_path="/PythonMeshManipulation/Figures/TunableReconstruction", before_after_plots=True)

        # lidar_img = y = P(i) @ R(0) @ Tvelo_cam @ x
