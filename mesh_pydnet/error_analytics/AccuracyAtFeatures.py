"""
Little experiment to determine which feature extraction will give the best depth accuracy on the pydnet
"""

import reconstruction.mesh_pydnet.error_analytics.ErrorEvaluationPydnetImgs as eeim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utilities.ImgFeatureExtactorModule as imFt

orb = imFt.FeatureDetector(det_type="orb", max_num_ft=2000)
sift = imFt.FeatureDetector(det_type="sift", max_num_ft=2000)


def orb_error(dense_ground_truth, prediction, rgb_image, MSE=True):
    """
    Gets the orb points in the rgb image and compares this to the dense ground truth
    Uses MSE if needed
    Returns Error
    """
    kps, des = orb.detect(rgb_image)
    uv = orb.kp_to_pts(kps)
    u, v = np.floor(uv).astype(int).T

    pred_disp = prediction[v,u]

    return eeim.get_img_pt_to_pt_error(dense_ground_truth, uv=uv, d=pred_disp, use_MSE=MSE)


def sift_error(dense_ground_truth, prediction, rgb_image, MSE=True):
    """
    Gets the SIFT points in the rgb image and compares this to the dense ground truth
    Uses MSE if needed
    Returns Error
    """
    kps, des = sift.detect(rgb_image)
    uv = sift.kp_to_pts(kps)
    u, v = np.floor(uv).astype(int).T

    pred_disp = prediction[v, u]

    return eeim.get_img_pt_to_pt_error(dense_ground_truth, uv=uv, d=pred_disp, use_MSE=MSE)


# def test_moc():
#     # TODO get moc features and check error at those points
#     pass


if __name__ == "__main__":
    import utilities.HuskyDataHandler as husky
    import reconstruction.VOComparison.aru_visual_odometry as avo
    import reconstruction.HuskyCalib as HuskyCalib
    import reconstruction.TunableReconstruction.Functions_TunableReconstruction as TR_func
    import os
    import tqdm

    data_dir = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/RouteC/2022_05_03_14_09_01"
    output_directory = os.path.join(data_dir, 'predictions/output/image_00/data')
    input_directory = os.path.join(data_dir, 'predictions/input/image_00/data')
    disp_directory = os.path.join(data_dir, 'predictions/disparity/image_00/data')
    dataset = husky.DatasetHandler(data_dir)
    depth_filenames, depth_times = avo.get_depth_predictions(disp_directory, dataset)

    """
    for each frame:
        - get dense-ified lidar
        - load predicted depth
        
        - check orb accuracy
        - check sift accuracy
        - check moc accuracy
        - check random point accuracy
    """
    mse_sift = np.zeros(dataset.num_frames)
    mse_orb = np.zeros(dataset.num_frames)
    for i in tqdm.trange(dataset.num_frames):
        velo = dataset.get_lidar(i)
        img = dataset.get_cam0(i)
        pred = np.load(os.path.join(disp_directory, depth_filenames[i]))

        u, v, d = eeim.lidar_to_img_frame(velo, HuskyCalib.T_cam0_vel0, dataset.calib.cam0_camera_matrix,
                                          img_shape=img.shape)
        # converting lidar from depth to disparity to work in the same space.
        d = TR_func.depth_to_disparity(d, K=dataset.calib.cam0_camera_matrix, t=dataset.calib.baseline)

        sparse_velo = eeim.render_lidar(np.stack((u, v, d), axis=1), Tr=HuskyCalib.T_cam0_vel0,
                                        K=dataset.calib.cam0_camera_matrix)
        rough_velo = eeim.rough_lidar_render(np.stack((u, v, d), axis=1),mask=True)

        mse_sift[i] = sift_error(dense_ground_truth=rough_velo, prediction=pred, rgb_image=img)
        mse_orb[i] = orb_error(dense_ground_truth=rough_velo, prediction=pred, rgb_image=img)

    plt.figure()
    plt.title("Errors between prediction and lidar")
    plt.plot(mse_orb, 'o-', label=f"MSE ORB Points (avg: {mse_orb.mean():.3f})")
    plt.plot(mse_sift, 'o-', label=f"MSE sift Points (avg: {mse_sift.mean():.3f})")
    plt.xlabel("Frame number")
    plt.ylabel("MSE at frame")
    plt.legend()

    plt.show()
