"""
This script compares the accuracy of visual odometry using stereo images vs using monocular depth predictions.
This gives an understanding of whether the monocular network is accurate enough for geometric reconstruction tasks.

Authur: Michael Katsoulis

"""
import reconstruction.aru_core_lib.aru_py_vo as aru_py_vo
import os
import numpy as np
import cv2
import rosbag
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm
import utilities.ImgFeatureExtactorModule as ft
import reconstruction.ModularFiles.MaximumOfCutvatureFeature as moc

from reconstruction.TunableReconstruction.Functions_TunableReconstruction import depth_to_disparity
from reconstruction.TunableReconstruction.Functions_TunableReconstruction import disparity_to_depth
import utilities.Transform as Transform
import utilities.Image as img_utils

# Temp:
import utilities.image_similarity as im_sim

sift = ft.FeatureDetector(det_type='sift', max_num_ft=1000)
orb = ft.FeatureDetector(det_type='orb', max_num_ft=1000)
pyvo = aru_py_vo.PyVO(
    '/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/VOComparison/config/vo_config.yaml')


def read_predictions(prediction_path):
    pass


def get_vo(frame0, frame1, data, pyvo=pyvo):
    stereo_left_n = cv2.cvtColor(data.get_cam0(frame0), cv2.COLOR_BGR2GRAY)
    stereo_right_n = cv2.cvtColor(data.get_cam1(frame0), cv2.COLOR_BGR2GRAY)
    stereo_left_np1 = cv2.cvtColor(data.get_cam0(frame1), cv2.COLOR_BGR2GRAY)
    stereo_right_np1 = cv2.cvtColor(data.get_cam1(frame1), cv2.COLOR_BGR2GRAY)

    t1 = data.left_times[frame1].astype(np.int64)
    t0 = data.left_times[frame0].astype(np.int64)

    # # TODO remove when done:
    # print()
    # print('#' * 60)
    # print(f"Left to Right SSIM for current frame: {im_sim.ssim(stereo_left_n, stereo_right_n):.4f}")
    # print(f"Left to Right SSIM for next frame: {im_sim.ssim(stereo_left_np1, stereo_right_np1):.4f}")
    # ll_ssim = im_sim.ssim(stereo_left_n, stereo_left_np1)
    # print(f"Left current frame to Left next frame: {ll_ssim:.4f}")
    # rr_ssim = im_sim.ssim(stereo_right_n, stereo_right_np1)
    # print(f"Right current frame to Right next frame: {rr_ssim:.4f}")
    # print('#' * 60)
    # print()
    #
    # if ll_ssim < 0.15 or rr_ssim < 0.15:
    #     cv2.imshow("Left n and np1", cv2.resize(np.hstack((stereo_left_n, stereo_left_np1)),
    #                                             (stereo_left_np1.shape[1], stereo_left_np1.shape[0] // 2)))
    #     cv2.imshow("Right n and np1", cv2.resize(np.hstack((stereo_right_n, stereo_right_np1)),
    #                                              (stereo_left_np1.shape[1], stereo_left_np1.shape[0] // 2)))
    #     cv2.waitKey(0)

    # cv2.imshow("Left n", stereo_left_n)
    # cv2.imshow("Left np1", stereo_left_np1)
    # cv2.imshow("Right n", stereo_right_n)
    # cv2.imshow("Right np1", stereo_right_np1)
    #
    # cv2.waitKey(0)
    return pyvo.stereo_odometry(stereo_left_n, stereo_right_n, t0, stereo_left_np1, stereo_right_np1, t1)


def generate_transform_path(transforms: np.array, validate: bool = False, max_d=1):
    coords = np.eye(4)
    x_vals = []
    y_vals = []

    print(f"Number of transforms with a nan or none:  {np.sum(np.isnan(transforms).sum(axis=-1).sum(axis=-1) > 0)}")
    for rt in transforms:
        if np.sum(np.isnan(rt)) == 0:
            if validate:
                dist, _ = Transform.distance_and_yaw_from_transform(rt)
                if dist < max_d:
                    coords = coords @ rt
                    # z-axis is fwd for img coords, x-axis is fwd in body coord sys
                    x_vals.append(coords[2][3])
                    # x-axis is right img coords, y-axis is left in body coord sys
                    y_vals.append(-coords[0][3])
            else:
                coords = coords @ rt
                # z-axis is fwd for img coords, x-axis is fwd in body coord sys
                x_vals.append(coords[2][3])
                # x-axis is right img coords, y-axis is left in body coord sys
                y_vals.append(-coords[0][3])
        # else:
        #     print("Found a None")

    return np.array(x_vals), np.array(y_vals)


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


def get_img_ft_pts(img_f0, img_f1, det=orb, lowes_ratio=0.7):
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


def get_num_similar_fts(im0, im1, use_sift=True, lowes_ratio=0.8):
    if use_sift:
        kps0, des0 = sift.detect(im0)
        kps1, des1 = sift.detect(im1)
        return len(sift.get_matches(des0, des1, lowes_ratio=lowes_ratio))
    else:
        kps0, des0 = orb.detect(im0)
        kps1, des1 = orb.detect(im1)
        return len(orb.get_matches(des0, des1, lowes_ratio=lowes_ratio))


def predicted_depth_motion_est(depth_f0, img_f0, img_f1, K, det=orb, min_matches=50, vo=pyvo, verify_trans=True,
                               max_step_d=2):
    """
    Calculate the estimated motion for a given depth image
    """

    uv0, uv1 = get_img_ft_pts(img_f0, img_f1, det=det)
    u0, v0 = uv0.T

    # Extracting camera parameters
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # getting 3d points in frame 0
    pt_u0 = np.floor(u0 * depth_f0.shape[1] / img_f0.shape[1]).astype(int)
    pt_v0 = np.floor(v0 * depth_f0.shape[0] / img_f0.shape[0]).astype(int)
    x_cam = (u0 - cx) * depth_f0[pt_v0, pt_u0] / fx
    y_cam = (v0 - cy) * depth_f0[pt_v0, pt_u0] / fy
    z_cam = depth_f0[pt_v0, pt_u0]

    # world_x = z_cam
    # world_y = -x_cam
    # world_z = -y_cam

    # pts3d = np.stack((world_x, world_y, world_z), axis=1)
    pts3d = np.stack((x_cam, y_cam, z_cam), axis=1)

    if len(uv1) < min_matches:
        return None

    if not verify_trans:
        return vo.motion_estimation(pts3d, uv1)
    t = vo.motion_estimation(pts3d, uv1)
    dist, deg = Transform.distance_and_yaw_from_transform(t)
    if dist > max_step_d:  # only append if moved less than 2m. at 0.1s per frame, that's faster than ...
        return None

    return t


def get_depth_vo_path_from_saved(output_directory, dataset, det=orb, start_frame=None, stop_frame=None,
                                 verify_trans=True, max_step_d=2):
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = dataset.num_frames - 1

    depth_filenames, depth_times = get_depth_predictions(output_directory, dataset)
    transforms = np.zeros((stop_frame - start_frame, 4, 4))

    for i in tqdm.trange(start_frame, stop_frame):
        d = depth_filenames[i]
        im0 = dataset.get_cam0(i)
        im1 = dataset.get_cam0(i + 1)
        d = np.load(os.path.join(output_directory, d))

        disp = depth_to_disparity(d, dataset.calib.cam0_camera_matrix, dataset.calib.baseline)
        if disp.shape != im0.shape[:2]:
            disp = disp * im0.shape[1] / disp.shape[1]
            disp = cv2.resize(disp, im0.shape[:2][::-1])
        depth = disparity_to_depth(disp, dataset.calib.cam0_camera_matrix, dataset.calib.baseline)

        K = dataset.calib.cam0_camera_matrix

        transforms[i - start_frame] = predicted_depth_motion_est(depth, img_f0=im0, img_f1=im1, K=K, det=det,
                                                                 verify_trans=verify_trans, max_step_d=max_step_d)

    return generate_transform_path(transforms)


def get_vo_path_on_dataset(dataset, start_frame=None, stop_frame=None, validate=False):
    if start_frame is None:
        start_frame = 0

    if stop_frame is None:
        stop_frame = dataset.num_frames - 1

    transforms = np.zeros((stop_frame - start_frame, 4, 4))

    for i in tqdm.trange(start_frame, stop_frame):
        transforms[i - start_frame] = get_vo(i, i + 1, dataset)

    return generate_transform_path(transforms, validate=validate)


def vo_on_bag(bag, max_d_per_step=2, stereo_cam_params: img_utils.StereoCamParams = None,
              stereo_topic="/camera/image_stereo/image_raw", left_topic='/camera/image_left/image_raw',
              right_topic='/camera/image_right/image_raw'):
    if stereo_cam_params is None:
        print("Input location of camera params")
        stereo_cam_params = img_utils.StereoCamParams(
            config_path="/home/kats/Code/aru_sil_py/config/aru-calibration/ZED")
        # TODO remove hard code
    topics = bag.get_type_and_topic_info()[1]
    if stereo_topic in topics:
        stereo_flag = True
    else:
        stereo_flag = False

    coords = np.eye(4)
    x_vals, y_vals = [], []
    if stereo_flag:
        num_imgs = bag.get_message_count(stereo_topic)
        img_gen = bag.read_messages(topics=stereo_topic)
        topic, msg, t_n = next(img_gen)
        bag_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        raw_left_im_n = bag_img[:msg.height, :msg.width // 2]
        raw_right_im_n = bag_img[:msg.height, msg.width // 2:]
    else:
        num_imgs = bag.get_message_count(left_topic)
        left_img_gen = bag.read_messages(topics=left_topic)
        right_img_gen = bag.read_messages(topics=right_topic)
        l_topic, l_msg, t_n = next(left_img_gen)
        r_topic, r_msg, r_t_n = next(right_img_gen)
        raw_left_im_n = np.frombuffer(l_msg.data, dtype=np.uint8).reshape(l_msg.height, l_msg.width, -1)
        raw_right_im_n = np.frombuffer(r_msg.data, dtype=np.uint8).reshape(r_msg.height, r_msg.width, -1)

    # Rectifying
    stereo_left_n = img_utils.rectify_image(cam_params=stereo_cam_params.left, image=raw_left_im_n)
    stereo_right_n = img_utils.rectify_image(cam_params=stereo_cam_params.right, image=raw_right_im_n)
    stereo_left_n = cv2.cvtColor(stereo_left_n, cv2.COLOR_BGR2GRAY)
    stereo_right_n = cv2.cvtColor(stereo_right_n, cv2.COLOR_BGR2GRAY)

    for i in tqdm.trange(0, num_imgs - 1):
        print()
        if stereo_flag:
            topic, msg_np1, t_np1 = next(img_gen)
            bag_img_np1 = np.frombuffer(msg_np1.data, dtype=np.uint8).reshape(msg_np1.height, msg_np1.width, -1)
            raw_left_im_np1 = bag_img_np1[:msg_np1.height, :msg_np1.width // 2]
            raw_right_im_np1 = bag_img_np1[:msg_np1.height, msg_np1.width // 2:]
        else:
            l_topic, l_msg_np1, t_np1 = next(left_img_gen)
            r_topic, r_msg_np1, r_t_np1 = next(right_img_gen)
            raw_left_im_np1 = np.frombuffer(l_msg_np1.data, dtype=np.uint8).reshape(l_msg_np1.height, l_msg_np1.width,
                                                                                    -1)
            raw_right_im_np1 = np.frombuffer(r_msg_np1.data, dtype=np.uint8).reshape(r_msg_np1.height, r_msg_np1.width,
                                                                                     -1)

        # Rectifying
        stereo_left_np1 = img_utils.rectify_image(cam_params=stereo_cam_params.left, image=raw_left_im_np1)
        stereo_right_np1 = img_utils.rectify_image(cam_params=stereo_cam_params.right, image=raw_right_im_np1)

        # cv2.imshow("Image left n", stereo_left_n)
        # cv2.imshow("Image right n", stereo_right_n)
        # cv2.imshow("Image left np1", stereo_left_np1)
        # cv2.imshow("Image right np1", stereo_right_np1)
        # cv2.waitKey(0)

        stereo_left_np1 = cv2.cvtColor(stereo_left_np1, cv2.COLOR_BGR2GRAY)
        stereo_right_np1 = cv2.cvtColor(stereo_right_np1, cv2.COLOR_BGR2GRAY)

        trans = pyvo.stereo_odometry(stereo_left_n, stereo_right_n, t_n.to_nsec(), stereo_left_np1, stereo_right_np1,
                                     t_np1.to_nsec())
        # print()
        # print(trans)

        # for the next loop
        stereo_left_n = stereo_left_np1
        stereo_right_n = stereo_right_np1
        t_n = t_np1

        # Validating
        dist, deg = Transform.distance_and_yaw_from_transform(trans)
        if dist < max_d_per_step:
            coords = coords @ trans
            # z-axis is fwd for img coords, x-axis is fwd in body coord sys
            x_vals.append(coords[2][3])
            # x-axis is right img coords, y-axis is left in body coord sys
            y_vals.append(-coords[0][3])
        else:
            print(f"Invalid transform at frame {i} \twith dist {dist:.2f}")
    return x_vals, y_vals


if __name__ == "__main__":
    import utilities.HuskyDataHandler as husky
    import matplotlib.pyplot as plt
    import utilities.plotting_utils as plt_utils


    # # Route A
    list_of_datasets = ["/home/kats/Datasets/Route A/2022_05_03_13_53_38"]
    list_of_bagfiles = ["/home/kats/Datasets/Route A/2022_05_03_13_53_38/bag/2022-05-03-13-53-38.bag"]


    for bagpath, dataset_path in zip(list_of_bagfiles, list_of_datasets):
        with rosbag.Bag(bagpath) as bag:
            x_vo_bag, y_vo_bag = vo_on_bag(bag)

        dataset_obj = husky.DatasetHandler(dataset_path)
        # x_vo_dset, y_vo_dset = get_vo_path(dataset_obj, start_frame=100, stop_frame=500, validate=True)
        x_vo_dset, y_vo_dset = get_vo_path_on_dataset(dataset_obj, validate=True)

        # Plots
        ############################################################################################
        fig, ax = plt.subplots()
        plt_utils.plot_vo_path_with_arrows(axis=ax, x=x_vo_bag, y=y_vo_bag, label="VO using the bag")
        plt_utils.plot_vo_path_with_arrows(axis=ax, x=x_vo_dset, y=y_vo_dset, label="VO using dataset")

        ax.legend()
        ax.axis('equal')

        fig.suptitle(f"Whitelab VO for file: {bagpath.rsplit('/')[-1]}")
        plt.show()



    # # Whitelab experiment:
    # list_of_bagfiles = ["/home/kats/Datasets/Whitelab/Bags/2022-07-04-10-25-03.bag",
    #                     "/home/kats/Datasets/Whitelab/Bags/2022-07-04-10-26-41.bag",
    #                     "/home/kats/Datasets/Whitelab/Bags/2022-07-04-10-28-32.bag"]
    # list_of_datasets = ["/home/kats/Datasets/Whitelab/Dataset_Structures/2022_07_04_10_25_03",
    #                     "/home/kats/Datasets/Whitelab/Dataset_Structures/2022_07_04_10_26_41",
    #                     "/home/kats/Datasets/Whitelab/Dataset_Structures/2022_07_04_10_28_32"]
    # # bagpath = "/home/kats/Datasets/Whitelab/Bags/2022-07-04-10-26-41.bag"
    # # dataset_path = "/home/kats/Datasets/Whitelab/Dataset_Structures/2022_07_04_10_26_41"
    #
    #
    # for bagpath, dataset_path in zip(list_of_bagfiles, list_of_datasets):
    #     with rosbag.Bag(bagpath) as bag:
    #         x_vo_bag, y_vo_bag = vo_on_bag(bag)
    #
    #     dataset_obj = husky.DatasetHandler(dataset_path)
    #     # x_vo_dset, y_vo_dset = get_vo_path(dataset_obj, start_frame=100, stop_frame=500, validate=True)
    #     x_vo_dset, y_vo_dset = get_vo_path_on_dataset(dataset_obj, validate=True)
    #
    #     # Plots
    #     ############################################################################################
    #     fig, ax = plt.subplots()
    #     plt_utils.plot_vo_path_with_arrows(axis=ax, x=x_vo_bag, y=y_vo_bag, label="VO using the bag")
    #     plt_utils.plot_vo_path_with_arrows(axis=ax, x=x_vo_dset, y=y_vo_dset, label="VO using dataset")
    #
    #     ax.legend()
    #     ax.axis('equal')
    #
    #     fig.suptitle(f"Whitelab VO for file: {bagpath.rsplit('/')[-1]}")
    #     plt.show()
    #
    # ###################################################################################

    # # data_dir = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/RouteC/2022_05_03_14_09_01"
    # # data_dir = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route A/2022_05_03_13_53_38" # Sunny Route A
    # data_dir = "/home/kats/Datasets/Route A/2022_05_03_13_53_38"
    #
    # output_directory = os.path.join(data_dir, 'predictions/output/image_00/data')
    # input_directory = os.path.join(data_dir, 'predictions/input/image_00/data')
    # dataset = husky.DatasetHandler(data_dir)
    # depth_filenames, depth_times = get_depth_predictions(output_directory, dataset)
    #
    # # ---------------------
    # # ---------------------
    #
    # video = False
    # plots = False
    # # start_frame = 45
    # start_frame =50
    # # end_frame = len(depth_times) -1
    # num_frames = 1000
    #
    # # if video:
    # #     video_output_directory = r"/home/kats/Videos/Masters/Depth_VO"
    # #     fps = 2
    # #     # out = cv2.VideoWriter(os.path.join(video_output_directory,'outpy.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
    # #     #                      fps, (1500, 700))
    # #     out = cv2.VideoWriter(os.path.join(video_output_directory, 'out_orb_2000fts.mp4'),
    # #                           cv2.VideoWriter_fourcc(*'mp4v'), fps, (1500, 700))
    #
    # print("Getting VO path")
    # # x_vo, y_vo = get_vo_path(dataset=dataset, start_frame=start_frame, stop_frame=end_frame)
    #
    # print("Getting pydnet orb path")
    # x_p, y_p = get_depth_vo_path_from_saved(output_directory=output_directory, dataset=dataset, det=orb, start_frame=start_frame,
    #                                         stop_frame=end_frame, verify_trans=True)
    #
    # print("Getting pydnet sift path")
    # x_psift, y_psift = get_depth_vo_path_from_saved(output_directory=output_directory, dataset=dataset, det=sift, start_frame=start_frame,
    #                                                 stop_frame=end_frame, verify_trans=True)
    #
    # # fixing coordinates from bindings
    # x_p = -x_p
    # x_psift = -x_psift
    #
    #
    # ###################################################################################################################
    # # Visualisation
    # ###################################################################################################################
    #
    #
    # print("importing matplotlib")
    # import matplotlib.pyplot as plt
    #
    # fig_pred, axp = plt.subplots(2, 2, figsize=(15, 7))
    # im0 = dataset.get_cam0(end_frame)
    #
    # axp[0, 0].imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
    # axp[0, 0].set_title(f"Image at frame {end_frame}")
    # axp[0, 0].axis('off')
    #
    # depth = np.load(os.path.join(output_directory, depth_filenames[end_frame]))
    # sc = axp[0, 1].imshow(depth, 'jet')
    # axp[0, 1].set_title(f" Predicted depth at frame {end_frame}")
    # axp[0, 1].axis('off')
    # divider = make_axes_locatable(axp[0, 1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cb = plt.colorbar(sc, cax=cax)
    # cb.set_label("Depth [m]")
    #
    # # es = t_p[:i + 1] - t_vo[:i  1]
    # # axp[1, 0].plot((es ** 2).mean(axis=-1).mean(axis=-1))
    # # axp[1, 0].set_title("MSE between VO and predicted depth VO transforms")
    #
    # # ds_vo = np.zeros(len(t_vo))
    # # for i in range(num_frames): ds_vo[i] = transform.distance_and_yaw_from_transform(t)[0]
    # # axp[1, 0].plot(ds_vo, '^-b', label='VO distances', markersize=2)
    # # ds_p = np.zeros(len(t_p))
    # # for i, t in enumerate(t_p): ds_p[i] = transform.distance_and_yaw_from_transform(t)[0]
    # # axp[1, 0].plot(ds_p, 'o-r', label='ORB Depth Pred distances', markersize=2)
    # # ds_psift = np.zeros(len(t_psift))
    # # for i, t in enumerate(t_psift): ds_psift[i] = transform.distance_and_yaw_from_transform(t)[0]
    # # axp[1, 0].plot(ds_psift, 'x-g', label='SIFT Depth Pred distances', markersize=2)
    # # axp[1, 0].legend()
    # # axp[1, 0].set_title("Comparison of distances from transforms")
    # # axp[1, 0].set_xlabel("Frame")
    # # axp[1, 0].set_ylabel("Meters")
    #
    # # axp[1, 1].plot(y_p, x_p, 'o--r', label='ORB Depth Pred Transforms', markersize=2)
    # # axp[1, 1].plot(y_psift, x_psift, 'x-g', label='SIFT Depth Pred Transforms', markersize=2)
    # # axp[1, 1].plot(y_vo, x_vo, '^--b', label='VO Transforms', markersize=2)
    # axp[1, 1].plot(y_p, x_p, 'o--r', label='ORB Depth Pred Transforms', markersize=2)
    # axp[1, 1].plot(y_psift, x_psift, 'x-g', label='SIFT Depth Pred Transforms', markersize=2)
    # axp[1, 1].plot(y_vo, x_vo, '^--b', label='VO Transforms', markersize=2)
    # axp[1, 1].axis('equal')
    # axp[1, 1].legend()
    # axp[1, 1].set_xlabel('x[m]')
    # axp[1, 1].set_ylabel('y[m]')
    #
    # plt.show()
    #
    # plt.figure()
    # plt.plot(x_psift, y_psift, label='Depth Pred Transforms')
    # plt.plot(x_vo, y_vo, label='VO Transforms')
    # plt.legend()
    # plt.axis('equal')
    # plt.xlabel('x[m]')
    # plt.ylabel('y[m]')
    # plt.show()
