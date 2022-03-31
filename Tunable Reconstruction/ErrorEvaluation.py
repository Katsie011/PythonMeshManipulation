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
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from HyperParameters import *
from matplotlib import cm

sys.path.insert(0, '/home/kats/Documents/Repos/aru-core/build/lib/')
import aru_py_mesh

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
#       Globals
# ----------------------------------------------------------------------------

# config_path = "/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/Tunable " \
# "Reconstruction/mesh_depth.yaml "
config_path = "mesh_depth.yaml"

depth_est = aru_py_mesh.PyDepth(config_path)


def make_sparse_dense(sparse_img, get_colour=False, mask=True):
    colour, depth = depth_est.create_dense_depth(sparse_img)

    if mask:
        depth = np.ma.array(depth, mask=(depth == -1))
        if get_colour:
            colour = np.ma.array(colour, mask=(colour == -1))

    if get_colour:
        return colour, depth
    else:
        return depth


def pts_to_img(pts, imshape=(375, 1242), dtype=np.uint8):
    im = np.zeros(imshape, dtype=dtype)
    idx = np.floor(pts[:, :2].T).astype(int)
    im[idx[1], idx[0]] = pts[:, 2]
    return im


def render_lidar(pts, img_shape=IMAGE_SHAPE, T_cam_velo=data.calib.T_cam0_velo, Rrect=data.calib.R_rect_00,
                 Prect=data.calib.P_rect_00, return_pts=False):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
       numpy array. Returns the 2D projection of the points that
       are in front of the camera only an the corresponding 3D points.'''
    # 3D points in camera reference frame.
    pts = pts.T
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts))
    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2, :] >= 0)
    pts2d_cam = Prect.dot(pts3d_cam[:, idx])
    pts3d = pts3d_cam[:, idx]
    pts2d_cam = pts2d_cam / pts2d_cam[2, :]
    valid = (pts2d_cam[0] < img_shape[1]) * (pts2d_cam[1] < img_shape[0]) * (pts2d_cam[0] > 0) * (pts2d_cam[1] > 0)
    p = np.hstack((pts2d_cam[:2, valid].T, pts3d[2, valid].reshape((-1, 1)))).T

    velo_im = np.zeros(img_shape)
    velo_im[np.floor(p[1]).astype(int), np.floor(p[0]).astype(int)] = p[2]

    if return_pts:
        return p.T
    else:
        return velo_im


def quantify_mesh_error_lidar(test_mesh_pts,
                              ground_truth_3d, gt_is_sparse_img=False, use_MSE=True,
                              generate_plots=False, normalise_by_test_pts=True):
    r"""
    Input mesh points and Ground truth
        - Ground truth can be sparse image or lidar points

    Will get Delaunay triangulation of input.
    Densify sparse Ground Truth
    Compare the two and give a quantitative output for error

    Choice of error metric: Sum of Squared Error (SSE), Mean Squared Error (MSE)

    Returns: Error
    """
    test_img = pts_to_img(test_mesh_pts)
    if gt_is_sparse_img:
        dense_depth = make_sparse_dense(ground_truth_3d)
    else:
        dense_depth = make_sparse_dense(pts_to_img(ground_truth_3d))

    dense_test = make_sparse_dense(test_img)

    e_im = (dense_depth - dense_test) ** 2

    if use_MSE:
        e_im = (dense_depth - dense_test) ** 2
        e = e_im.sum() / e_im.count()
    else:
        e = np.sum(e_im)

    return e


def error_heatmap(mesh_pts, ground_truth_3D, gt_is_sparse_img=False, normalise_by_mean=False, img_shape=IMAGE_SHAPE,
                  colourmap=plt.get_cmap('jet')):
    r"""
    Returns Heat map of error
    """
    test_img = pts_to_img(mesh_pts)
    dense_test = make_sparse_dense(test_img)
    if gt_is_sparse_img:
        gt_im = make_sparse_dense(ground_truth_3D)
    else:
        gt_im = make_sparse_dense(pts_to_img(ground_truth_3D))

    emap = ((gt_im - dense_test) ** 2)

    if normalise_by_mean:
        emap = emap / emap.mean()

    return colourmap(emap)


if __name__ == "__main__":
    import IterativeTunableReconstructionPipeline as it_recon

    frame = 0
    im0 = np.array(data.get_cam0(0))
    im1 = np.array(data.get_cam1(0))
    mesh, pts = it_recon.iterative_recon(im0, im1, before_after_plots=True)

    import matplotlib.pyplot as plt

    velo = render_lidar(data.get_velo(0))
    p = render_lidar(data.get_velo(0), return_pts=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.7))
    fig.tight_layout()
    ax.imshow(im0, 'gray')
    plt.imshow(make_sparse_dense(pts_to_img(pts)))
    fig.suptitle("Iterative Tunable Reconstruction Dense Interpolation")
    ax.axis('off')
    plt.show()

    new_p = p[::30]
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.7))
    fig.tight_layout()
    ax.imshow(im0, 'gray')
    plt.imshow(make_sparse_dense(pts_to_img(new_p)))
    fig.suptitle("Ground Truth Dense Interpolation")
    ax.axis('off')
    plt.show()

    print(f"Error is: {quantify_mesh_error_lidar(pts, new_p, gt_is_sparse_img=False)}")

    print("Getting Error Map")
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.7))
    fig.tight_layout()
    ax.imshow(im0, 'gray')
    ax.imshow(error_heatmap(pts, pts_to_img(new_p), gt_is_sparse_img=True))
    fig.suptitle("Error Map", fontsize=18)
    ax.axis('off')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.7))
    fig.tight_layout()
    ax.imshow(im0, 'gray')
    it_recon.plot_mesh(mesh, pts, a=ax)
    fig.suptitle("Iterative Tunable Reconstruction Results")
    ax.axis('off')
    plt.show()
