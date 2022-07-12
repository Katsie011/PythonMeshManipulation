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
import utilities.HuskyDataHandler as husky

# Temp:
import utilities.image_similarity as im_sim

sift = ft.FeatureDetector(det_type='sift', max_num_ft=2000)
orb = ft.FeatureDetector(det_type='orb', max_num_ft=2000)
pyvo = aru_py_vo.PyVO(
    '/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/VOComparison/config/vo_config.yaml')




def get_stereo_disp(kpsl, desl, kpsr, desr, det=sift, lowes_ratio: float = 0.9):
    good_matches = det.get_matches(desl, desr, lowes_ratio=lowes_ratio)
    idxl, idxr = det.matches_and_keypts_to_good_idxs(good_matches, kps_l=kpsl, kps_r=kpsr)
    kpsl=kpsl[idxl]
    kpsr = kpsr[idxr]
    pts_l = det.kp_to_pts(kpsl)
    pts_r = det.kp_to_pts(kpsr)

    # train_idxs = np.zeros(len(good_matches), dtype=int)
    # query_idxs = np.zeros(len(good_matches), dtype=int)
    #
    # for i, m in enumerate(good_matches):
    #     train_idxs[i] = m.trainIdx
    #     query_idxs[i] = m.queryIdx
    #
    # kpsl = np.array(kpsl, dtype=object)
    # kpsr = np.array(kpsr, dtype=object)
    #
    # pts_l = det.kp_to_pts(kpsl[query_idxs])
    # pts_r = det.kp_to_pts(kpsr[train_idxs])

    disp = (pts_l[:, 0] - pts_r[:, 0])
    return disp, pts_l, pts_r


def get_good_stereo_depth(kpsl, desl, kpsr, desr, dataset: husky.DatasetHandler, det=sift, use_disparity=True,
                          epipolar_window=2):
    disp, pts_l, pts_r = get_stereo_disp(kpsl, desl, kpsr, desr, det=det)
    good_disp_idx = (disp > 0) & (np.abs(pts_l[:, 1] - pts_r[:, 1]) < epipolar_window)
    good_pts_l = pts_l[good_disp_idx]
    good_pts_r = pts_r[good_disp_idx]
    good_disp = disp[good_disp_idx]

    if not use_disparity:
        d = dataset.calib.baseline[0] * (
            dataset.calib.cam0_camera_matrix[0, 0]) / good_disp
    else:
        d = good_disp
    return np.stack((good_pts_l[:, 0], good_pts_l[:, 1], d), axis=1)


def get_stereo_motion_est_vo(stereo_left_n, stereo_right_n, left_np1,
                             dataset_obj: husky.DatasetHandler,
                             det_obj=sift, epipolar_window: int = 2,
                             py_vo_obj: aru_py_vo.PyVO = pyvo, lowes_ratio: float = 0.9):
    # TODO finish this and get the vo using input stereo disparity
    kpsl0, desl0 = det_obj.detect(stereo_left_n)
    kpsl1, desl1 = det_obj.detect(left_np1)
    kpsr0, desr0 = det_obj.detect(stereo_right_n)

    # Getting the stereo depth and the indicies of the good matches
    # f0_stereo_pts = get_good_stereo_depth(kpsl0, desl0, kpsr0, desr0, dataset=dataset_obj,
    #                                                      use_disparity=False)


    # getting matches left to right frame 0
    good_matches = det_obj.get_matches(desl0, desr0, lowes_ratio=lowes_ratio)
    idx_l0_l_to_r, idx_r0_l_to_r = det_obj.matches_and_keypts_to_good_idxs(good_matches, kps_l=kpsl0, kps_r=kpsr0)
    matched_kpsl0 = np.array(kpsl0, dtype=object)[idx_l0_l_to_r]
    matched_kpsr = np.array(kpsr0, dtype=object)[idx_r0_l_to_r]
    pts_l_f0, pts_r_f0 =det_obj.kp_to_pts(matched_kpsl0), det_obj.kp_to_pts(matched_kpsr)
    disp = (pts_l_f0[:, 0] - pts_r_f0[:, 0])
    good_disp_idx = (disp > 0) & (np.abs(pts_l_f0[:, 1] - pts_r_f0[:, 1]) < epipolar_window)
    good_disp = disp[good_disp_idx]

    # matching the good points in the first stereo pair to the next frame
    matches_f0_f1 = det_obj.get_matches(desl0, desl1, lowes_ratio=lowes_ratio)
    idx_l0_f0to1, idx_l1_f0to1 = det_obj.matches_and_keypts_to_good_idxs(matches_f0_f1, kps_l=matched_kpsl0, kps_r=kpsl1)
    # matched_kpsl0 = np.array(matched_kpsr, dtype=object)[idx_l0_f0to1]
    matched_kpsl1 = np.array(kpsl1, dtype=object)[idx_l1_f0to1]


    # get the stereo depths that match to a point in the next frame:
    #   - First combine index selections
    idx = np.zeros(len(kpsl0), dtype=bool)
    idx_f0tof1 = idx.copy(); idx_f0tof1[idx_l0_f0to1] = True
    idx_l_to_r = idx.copy(); idx_l_to_r[idx_l0_l_to_r] = True
    valid_disp_pts = idx_f0tof1[idx_l_to_r]
    valid_good_disp = valid_disp_pts[good_disp_idx]
    matched_disp = good_disp[valid_good_disp]

    stereo_depths = dataset.calib.baseline[0] * (dataset.calib.cam0_camera_matrix[0, 0]) / matched_disp
    left_stereo_uv = pts_l_f0[np.logical_and(good_disp_idx, valid_disp_pts)]
    #       - Use this to select final stereo/3D points from frame 0
    pts3d = np.stack((left_stereo_uv[:,0], left_stereo_uv[:,1], stereo_depths), axis=1)

    # get the positions of the good points in the next frame:
    f1_uv_pts = det_obj.kp_to_pts(matched_kpsl1)

    assert f1_uv_pts.shape == pts3d.shape, "The shapes of the 3D points and the next frame uv points"

    #TODO The shapes arent matching
    # Need to check that the indices are working correctly
    # not sure what is right and what is not.
    return py_vo_obj.motion_estimation(pts3d, f1_uv_pts)







def get_vo(frame0, frame1, data, pyvo=pyvo):
    stereo_left_n = data.get_cam0(frame0)
    stereo_right_n = data.get_cam1(frame0)
    stereo_left_np1 = data.get_cam0(frame1)
    stereo_right_np1 = data.get_cam1(frame1)

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

    print(f"Number of transforms with a nan or None:  {np.sum(np.isnan(transforms).sum(axis=-1).sum(axis=-1) > 0)}")
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

    # TODO redundant. Use descripter matches

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


def motion_est_from_ft_pts(uv_f0, uv_f1, depth, K, min_matches=50, vo=pyvo, verify_trans=True,
                               max_step_d=2):
    """
    Calculate the estimated motion for a given depth image
    """
    # if len(uv1) < min_matches:
    #     return None
    pt_u0, pt_v0 = uv_f0.T.astype(int)

    # Extracting camera parameters
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # getting 3d points in frame 0
    # pt_u0 = np.floor(u0 * depth_f0.shape[1] / img_f0.shape[1]).astype(int)
    # pt_v0 = np.floor(v0 * depth_f0.shape[0] / img_f0.shape[0]).astype(int)
    x_cam = (pt_u0 - cx) * depth / fx
    y_cam = (pt_v0 - cy) * depth / fy
    z_cam = depth
    pts3d = np.stack((x_cam, y_cam, z_cam), axis=1)

    if not verify_trans:
        return vo.motion_estimation(pts3d, uv_f1)
    t = vo.motion_estimation(pts3d, uv_f1)
    dist, deg = Transform.distance_and_yaw_from_transform(t)
    if dist > max_step_d:  # only append if moved less than 2m. at 0.1s per frame, that's faster than ...
        return None
    return t

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


def vo_on_bag(bag: rosbag.Bag, max_d_per_step: int = 2, camera_config_path: str = None,
              stereo_topic="/camera/image_stereo/image_raw", left_topic='/camera/image_left/image_raw',
              right_topic='/camera/image_right/image_raw'):
    """
    Given a rosbag bag, runs VO on the bag and returns the path in global x and y [m]
    """
    if camera_config_path is None:
        stereo_cam_params = img_utils.StereoCamParams()
    else:
        stereo_cam_params = img_utils.StereoCamParams(config_path=camera_config_path)

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


if __name__=="__main__":
    dataset = husky.DatasetHandler("/home/kats/Datasets/Route A/2022_07_06_10_48_24")

    f=0
    print(get_stereo_motion_est_vo(dataset.get_cam0(f), dataset.get_cam1(f), dataset.get_cam0(f+1), dataset))