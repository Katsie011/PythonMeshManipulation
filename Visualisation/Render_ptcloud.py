import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import ModularFiles as husky
import TunableReconstruction.Functions_TunableReconstruction as TR_func
import TunableReconstruction.ErrorEvaluationImgs as ee
import mesh_pydnet.HuskyCalib as HuskyCalib


test_dir = "/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test_Route_C"
training_dir = "/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train_Route_C"
output_directory = os.path.join(test_dir, 'predictions')


dataset = husky.Dataset_Handler(test_dir)

points = dataset.get_lidar(0)

u, v, gt_depth = ee.lidar_to_img_frame(points, HuskyCalib.T_cam0_vel0, dataset.calib.cam0_camera_matrix,
                                       img_shape=dataset.calib.cam0_img_shape)
gt_disparity = TR_func.depth_to_disparity(gt_depth, dataset.calib.cam0_camera_matrix, dataset.calib.baseline)
velo_cam = np.floor(np.stack((u, v, gt_disparity), axis=1)).astype(int)
velo_cam_depth = np.stack((u, v, gt_depth), axis=1)

plt.imshow(dataset.get_cam0(0))
plt.show()



"""
Depth image to point cloud:
z = d / depth_scale
x = (u - cx) * z / fx
y = (v - cy) * z / fy
"""

K = dataset.calib.cam0_camera_matrix
fx = K[0, 0]
fy = K[1, 1]
cx = K[0,2]
cy = K[1,2]

z = gt_depth
x = (u - cx) * z / fx
y = (-v + cy) * z / fy

pts = np.stack((x,y,z), axis=-1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors/65535)
# pcd.normals = o3d.utility.Vector3dVector(normals)
o3d.visualization.draw_geometries([pcd])
