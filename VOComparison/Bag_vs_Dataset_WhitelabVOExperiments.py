"""
Script for assessing the performance of VO between bagfiles and datasets.


"""

from reconstruction.VOComparison.aru_visual_odometry import *

import utilities.HuskyDataHandler as husky
import matplotlib.pyplot as plt
import utilities.plotting_utils as plt_utils
import argparse


parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--dataset_structure', type=str, help='Path to extracted dataset structures ',
                    default=r"/home/kats/Datasets/Whitelab/Dataset_Structures/2022_07_04_10_25_03")
parser.add_argument('--bagfile', type=str, help='path to the bagfile',
                    default=r"/home/kats/Datasets/Whitelab/Bags/2022-07-04-10-25-03.bag")
parser.add_argument('--camera_config_folder', type=str, help="Path to camera parameters folder",
                    default=r"/home/kats/Code/aru_sil_py/config/aru-calibration/ZED")
args = parser.parse_args()



if __name__ == "__main__":
    bagpath = args.bagfile
    dataset_path =args.dataset_structure

    with rosbag.Bag(bagpath) as bag:
        x_vo_bag, y_vo_bag = vo_on_bag(bag, camera_config_path=args.camera_config_folder)

    dataset_obj = husky.DatasetHandler(dataset_path)
    x_vo_dset, y_vo_dset = get_vo_path(dataset_obj, start_frame=100, stop_frame=500, validate=True)
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
