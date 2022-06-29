"""
This script compares the accuracy of visual odometry using stereo images vs using monocular depth predictions.
This gives an understanding of whether the monocular network is accurate enough for geometric reconstruction tasks.

Authur: Michael Katsoulis

"""
import reconstruction.aru_core_lib.aru_py_vo as aru_py_vo
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm
import utilities.HuskyDataHandler as husky
import utilities.ImgFeatureExtactorModule as ft
import reconstruction.ModularFiles.MaximumOfCutvatureFeature as moc

from reconstruction.TunableReconstruction.Functions_TunableReconstruction import depth_to_disparity
from reconstruction.TunableReconstruction.Functions_TunableReconstruction import disparity_to_depth

sift = ft.FeatureDetector(det_type='sift', max_num_ft=2000)
orb = ft.FeatureDetector(det_type='orb', max_num_ft=2000)
pyvo = aru_py_vo.PyVO(
    '/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/VOComparison/config/vo_config.yaml')


def read_predictions(prediction_path):
    pass


def get_vo(frame0, frame1, data, pyvo=pyvo):
    s0l = cv2.cvtColor(data.get_cam0(frame0, rectify=True), cv2.COLOR_BGR2GRAY)
    s0r = cv2.cvtColor(data.get_cam1(frame0, rectify=True), cv2.COLOR_BGR2GRAY)
    s1r = cv2.cvtColor(data.get_cam1(frame1, rectify=True), cv2.COLOR_BGR2GRAY)
    s1l = cv2.cvtColor(data.get_cam0(frame1, rectify=True), cv2.COLOR_BGR2GRAY)
    t1 = data.left_times[frame1].astype(np.int64)
    t0 = data.left_times[frame0].astype(np.int64)

    return pyvo.stereo_odometry(s0l, s0r, t0, s1l, s1r, t1)


def get_vo_path(dataset):
    transforms = np.zeros((dataset.num_frames - 1, 4, 4))

    # for i in tqdm.trange(dataset.num_frames-1):
    for i in tqdm.trange(500):
        transforms[i] = get_vo(i, i + 1, dataset)

    return generate_transform_path(transforms)


def generate_transform_path(transforms: np.array):
    coords = np.eye(4)
    x_vals = []
    y_vals = []
    for transform in transforms:
        coords = coords @ transform
        # z-axis is fwd for img coords, x-axis is fwd in body coord sys
        x_vals.append(coords[2][3])
        # x-axis is right img coords, y-axis is left in body coord sys
        y_vals.append(-coords[0][3])

    return [x_vals, y_vals]


def get_depth_predictions(predictions_directory, dataset):
    out_files = os.listdir(predictions_directory)

    unsynced_pred_files = np.asarray(os.listdir(predictions_directory), dtype=object)
    unsynced_times = -1 * np.ones(len(unsynced_pred_files))
    pred_files = np.zeros(len(dataset.left_image_files), dtype=object)
    pred_times = -1 * np.ones(dataset.num_frames)
    for i, f in enumerate(unsynced_pred_files):
        # Getting times
        unsynced_times[i] = int(os.path.splitext(f)[0])

    for i, lt in enumerate(dataset.left_times):
        # Need to get the first image where t_right > t_left
        # Right images are published after left but aquired simultaneously

        # getting the time difference w.r.t. current right time
        difference = unsynced_times - lt
        pred_ind = np.argmin(np.abs(difference))  # minimum along left times

        if difference[pred_ind] * dataset.timestamp_resolution < dataset.time_tolerance:
            # getting the first frame where right is after left (first positive time)
            pred_files[i] = unsynced_pred_files[pred_ind]
            pred_times[i] = unsynced_times[pred_ind]

            # need to reassure that this matches all files and that right images are not published after left.

            # removing the matched file from the right files that still need to be matched
            # this is to avoid duplicate matches
            ind = np.ones(len(unsynced_pred_files), dtype=bool)
            ind[pred_ind] = False

            unsynced_pred_files = unsynced_pred_files[ind]
            unsynced_times = unsynced_times[ind]

        else:
            print(f"Could not match {[dataset.left_image_files][i]} with a depth prediction file")

    return pred_files, pred_times

def get_orb_ft_pts(img_f0, img_f1, det=orb, lowes_ratio=0.7):
    """
    Returns the matched points for the two images.

    Input:
        img_0
        img_1

    Returns:
        uv_img0
        uv_img1
    """
    kps0, des0 = det.detect(img_f0)
    kps1, des1 = det.detect(img_f1)

    matches = det.get_matches(des0, des1, lowes_ratio=lowes_ratio)

    # Getting the indicies of the matches
    train_idx = []
    query_idx = []
    for m in matches:
        train_idx.append(m.trainIdx)
        query_idx.append(m.queryIdx)

    # Getting coordinates of the keypoints at the matches
    # print(f"Size K0 {len(kps0)}, train:{len(train_idx)}, max {np.max(train_idx)}")
    # print(f"Size K1 {len(kps1)}, query:{len(query_idx)}, max {np.max(query_idx)}")
    uv0 = det.kp_to_pts(kps0)[query_idx]
    uv1 = det.kp_to_pts(kps1)[train_idx]

    return uv0, uv1

def get_moc_ft_pts(img_f0, img_f1):
    """
    Returns the matched points for the two images using Maximum of Curvature

    Input:
        img_0
        img_1

    Returns:
        uv_img0
        uv_img1

    References:
        [1] M. Yokozuka, et al. "VITAMIN-E: VIsual Tracking And MappINg with Extremely Dense Feature Points",
        2019.
    """

    kappa = moc.curvature(img_f0 / 255.0)
    knorm = np.linalg.norm(kappa, axis=-1)
    max_msk, vu0 = moc.local_maxima(knorm, wsize=32)






def depth_pred_vo(depth, img_f0, img_f1, K, det=orb, min_matches=50, vo=pyvo, verify_trans = False, max_step_d=2):
    uv0, uv1 = get_orb_ft_pts(img_f0, img_f1, det=det)
    u0, v0 = uv0.T

    # Extracting camera parameters
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # getting 3d points in frame 0
    pt_u0 = np.floor(u0 * depth.shape[1] / img_f0.shape[1]).astype(int)
    pt_v0 = np.floor(v0 * depth.shape[0] / img_f0.shape[0]).astype(int)
    x_cam = (u0 - cx) * depth[pt_v0, pt_u0] / fx
    y_cam = (v0 - cy) * depth[pt_v0, pt_u0] / fy
    z_cam = depth[pt_v0, pt_u0]

    # world_x = z_cam
    # world_y = -x_cam
    # world_z = -y_cam

    # pts3d = np.stack((world_x, world_y, world_z), axis=1)
    pts3d = np.stack((x_cam, y_cam, z_cam), axis=1)

    # get VO estimation from 3d points in frame 0 and 2d pts in frame 1
    if not verify_trans:
        return vo.motion_estimation(pts3d, uv1)
    else:
        t = vo.motion_estimation(pts3d, uv1)
        dist, deg = transform.distance_and_yaw_from_transform(t)
        if dist > max_step_d:  # only append if moved less than 2m. at 0.1s per frame, that's faster than ...
            print("Distance ")
            for i in range(len(u0)):
                img_f0 = cv2.circle(img_f0, uv0[i].astype(int), radius=10, color=(100, 5, 254), thickness=4)
                img_f1 = cv2.circle(img_f1, uv1[i].astype(int), radius=10, color=(100, 5, 254), thickness=4)

            plt.figure()
            plt.tight_layout(3.)
            plt.title(f"Images with transform transform distance: {dist:.2f} and {len(uv0)} feature matches")
            plt.imshow(cv2.cvtColor(np.hstack((img_f0, img_f1)), cv2.COLOR_RGB2BGR))
            plt.show()

        return t




def get_depth_vo_path(output_directory, dataset):
    depth_filenames, depth_times = get_depth_predictions(output_directory, dataset)

    transforms = np.zeros((dataset.num_frames, 4, 4))

    # for i in tqdm.trange(dataset.num_frames-1):
    for i in tqdm.trange(500):
        d = depth_filenames[i]
        im0 = dataset.get_cam0(i)
        im1 = dataset.get_cam0(i + 1)
        d = cv2.imread(os.path.join(output_directory, d), cv2.IMREAD_GRAYSCALE)

        disp =depth_to_disparity(d, dataset.calib.cam0_camera_matrix, dataset.calib.baseline)
        if disp.shape != im0.shape[:2]:
            disp = disp * im0.shape[1] / disp.shape[1]
            disp = cv2.resize(disp, im0.shape[:2][::-1])
        depth =  disparity_to_depth(disp, dataset.calib.cam0_camera_matrix, dataset.calib.baseline)

        K = dataset.calib.cam0_camera_matrix

        transforms[i] = depth_pred_vo(depth, img_f0=im0, img_f1=im1, K=K)

    return generate_transform_path(transforms)


if __name__ == "__main__":
    import utilities.Transform as transform
    # data_dir = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/RouteC/2022_05_03_14_09_01"
    # data_dir = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route A/2022_05_03_13_53_38" # Sunny Route A
    data_dir = "/home/kats/Datasets/Route A/2022_05_03_13_53_38"

    output_directory = os.path.join(data_dir, 'predictions/output/image_00/data')
    input_directory = os.path.join(data_dir, 'predictions/input/image_00/data')
    dataset = husky.DatasetHandler(data_dir)
    depth_filenames, depth_times = get_depth_predictions(output_directory, dataset)


    # ---------------------
    # ---------------------


    video = False
    plots = False
    start_frame = 10
    num_frames = len(depth_times)
    # num_frames = 600

    # ---------------------
    # ---------------------
    # ---------------------
    # ---------------------
    # ---------------------
    # ---------------------


    # if video:
    #     video_output_directory = r"/home/kats/Videos/Masters/Depth_VO"
    #     fps = 2
    #     # out = cv2.VideoWriter(os.path.join(video_output_directory,'outpy.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1500, 700))
    #     out = cv2.VideoWriter(os.path.join(video_output_directory, 'out_orb_2000fts.mp4'),
    #                           cv2.VideoWriter_fourcc(*'mp4v'), fps, (1500, 700))



    x_vo, y_vo = [], []
    x_p, y_p = [], []
    x_psift, y_psift = [], []
    coords_p = np.eye(4)
    coords_psift = np.eye(4)
    coords_vo = np.eye(4)
    K = dataset.calib.cam0_camera_matrix

    t_p = np.zeros((num_frames, 4, 4))
    t_psift = np.zeros((num_frames, 4, 4))
    t_vo = np.zeros((num_frames, 4, 4))
    # for i in tqdm.trange(len(depth_times)-1):
    for i in tqdm.trange(start_frame,num_frames-2):
        # Show current image, future image and depth

        # Get transform
        d = depth_filenames[i]
        # depth = cv2.imread(os.path.join(output_directory, d), cv2.IMREAD_GRAYSCALE)
        depth = np.load(os.path.join(output_directory, d), cv2.IMREAD_GRAYSCALE)
        im0 = dataset.get_cam0(i)
        im1 = dataset.get_cam0(i + 1)

        # Getting transforms
        t_p[i] = depth_pred_vo(depth, img_f0=im0, img_f1=im1, K=K, verify_trans=True)
        t_psift[i] = depth_pred_vo(depth, img_f0=im0, img_f1=im1, K=K, det=sift, verify_trans=True)
        t_vo[i] = get_vo(i, i + 1, dataset)


        coords_p = coords_p @ t_p[i]
        x_p.append(-coords_p[2][3])
        y_p.append(coords_p[0][3])

        coords_psift = coords_psift @ t_psift[i]
        x_psift.append(-coords_psift[2][3])
        y_psift.append(coords_psift[0][3])

        coords_vo = coords_vo @ t_vo[i]
        x_vo.append(coords_vo[2][3])
        y_vo.append(-coords_vo[0][3])


    #     # Making the plots
    #     if plots:
    #         axp[0, 0].imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
    #         axp[0, 0].set_title(f"Image at frame{i}")
    #         axp[0, 0].axis('off')
    #
    #         # axp[0, 1].imshow(im1)
    #         # axp[0, 1].set_title(f"Image at frame{i+1}")
    #         # axp[0, 1].axis('off')
    #         #
    #         # sc = axp[1, 0].imshow(depth, 'jet')
    #         # axp[1, 0].set_title(f" Predicted depth at frame{i}")
    #         # axp[1, 0].axis('off')
    #         # divider = make_axes_locatable(axp[1, 0])
    #         # cax = divider.append_axes("right", size="5%", pad=0.05)
    #         # cb = plt.colorbar(sc, cax=cax)
    #         # cb.set_label("Depth [m]")
    #
    #         sc = axp[0, 1].imshow(depth, 'jet')
    #         axp[0, 1].set_title(f" Predicted depth at frame{i}")
    #         axp[0, 1].axis('off')
    #         divider = make_axes_locatable(axp[0, 1])
    #         cax = divider.append_axes("right", size="5%", pad=0.05)
    #         cb = plt.colorbar(sc, cax=cax)
    #         cb.set_label("Depth [m]")
    #
    #         es = t_p[:i + 1] - t_vo[:i + 1]
    #         axp[1, 0].plot((es ** 2).mean(axis=-1).mean(axis=-1))
    #         axp[1, 0].set_title("MSE between VO and predicted depth VO transforms")
    #         axp[1, 1].plot(x_p, y_p, 'o--r',label='Depth Pred Transforms')
    #         axp[1, 1].plot(x_vo, y_vo, '^--b', label='VO Transforms')
    #         axp[1, 1].axis('equal')
    #         if i ==0:
    #             axp[1, 1].legend()
    #         axp[1, 1].set_xlabel('x[m]')
    #         axp[1, 1].set_ylabel('y[m]')
    #
    #         # fig_pred.canvas.draw()
    #         # fig_pred.canvas.flush_events()
    #
    #         fig_pred.canvas.draw()
    #         canv = np.fromstring(fig_pred.canvas.tostring_rgb(), dtype=np.uint8,
    #                              sep='')
    #         canv = canv.reshape(fig_pred.canvas.get_width_height()[::-1] + (3,))
    #         canv = cv2.cvtColor(canv, cv2.COLOR_RGB2BGR)
    #         if video:
    #             out.write(canv)
    #         # plt.show()
    #         cv2.imshow('VO Plots', canv)
    #         cv2.waitKey(1)
    #
    #
    #         plt.close('all') # TODO Make the figures replot on the same figure/animate without making new figs
    #
    # if video: out.release()

    fig_pred, axp = plt.subplots(2, 2, figsize=(15, 7))

    axp[0, 0].imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
    axp[0, 0].set_title(f"Image at frame {i}")
    axp[0, 0].axis('off')

    sc = axp[0, 1].imshow(depth, 'jet')
    axp[0, 1].set_title(f" Predicted depth at frame {i}")
    axp[0, 1].axis('off')
    divider = make_axes_locatable(axp[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label("Depth [m]")

    # es = t_p[:i + 1] - t_vo[:i  1]
    # axp[1, 0].plot((es ** 2).mean(axis=-1).mean(axis=-1))
    # axp[1, 0].set_title("MSE between VO and predicted depth VO transforms")

    ds_vo = np.zeros(len(t_vo))
    for i,t in enumerate(t_vo): ds_vo[i] = transform.distance_and_yaw_from_transform(t)[0]
    axp[1,0].plot(ds_vo,'^-b', label='VO distances', markersize=2)
    ds_p = np.zeros(len(t_p))
    for i,t in enumerate(t_p): ds_p[i] = transform.distance_and_yaw_from_transform(t)[0]
    axp[1,0].plot(ds_p,'o-r', label='ORB Depth Pred distances', markersize=2)
    ds_psift = np.zeros(len(t_psift))
    for i,t in enumerate(t_psift): ds_psift[i] = transform.distance_and_yaw_from_transform(t)[0]
    axp[1,0].plot(ds_psift, 'x-g', label='SIFT Depth Pred distances', markersize=2)
    axp[1,0].legend()
    axp[1, 0].set_title("Comparison of distances from transforms")
    axp[1, 0].set_xlabel("Frame")
    axp[1, 0].set_ylabel("Meters")

    axp[1, 1].plot(y_p, x_p,  'o--r', label='ORB Depth Pred Transforms', markersize=2)
    axp[1, 1].plot(y_psift, x_psift,  '--g', label='SIFT Depth Pred Transforms', markersize=2)
    axp[1, 1].plot(y_vo, x_vo,  '^--b', label='VO Transforms', markersize=2)
    axp[1, 1].axis('equal')
    axp[1, 1].legend()
    axp[1, 1].set_xlabel('x[m]')
    axp[1, 1].set_ylabel('y[m]')

    plt.show()


    plt.figure()
    plt.plot(x_p, y_p, label='Depth Pred Transforms')
    plt.plot(x_vo, y_vo, label='VO Transforms')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.show()

