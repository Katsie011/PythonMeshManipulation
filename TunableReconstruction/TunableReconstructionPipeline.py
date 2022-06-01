r"""
Implementation of High Resolution Tunable Reconstruction from https://arxiv.org/pdf/1511.00758.pdf
Implemented by:                                                         Michael Katsoulis March 2022


This is the pipeline implementation after experimenting with the concept in a jupyter notebook.

"""

# -----------------------------------------------------------------------------------------------------------------
#       Importing libraries
# -----------------------------------------------------------------------------------------------------------------
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
# import aru_py_mesh
import sys

sys.path.insert(0, "/ModularFiles")
# import DatasetHandler
import ImgFeatureExtactorModule as ft
# import Kitti_Dataset_Files_Handler
from PythonMeshManipulation.mesh_pydnet.HyperParameters import *

# /home/kats/Documents/My Documents/Datasets/KITTI_cvlibs/2011_09_26/2011_09_26_drive_0001_sync
base = r"/home/kats/Documents/My Documents/Datasets/KITTI_cvlibs/"
date = "2011_09_26"
drive = "0001"

data = pykitti.raw(base, date, drive)

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



# -----------------------------------------------------------------------------------------------------------------
#       Extracting Features
# -----------------------------------------------------------------------------------------------------------------
print("Extracting features")
img0 = np.array(data.get_cam0(0))
img1 = np.array(data.get_cam1(0))

if DETECTOR == 'ORB':
    det = ft.FeatureDetector(det_type='orb', max_num_ft=1000)
else:
    det = ft.FeatureDetector(det_type='sift', max_num_ft=MAX_NUM_FEATURES_DETECT)
# kp0 = det.detect(img0)
# kp1 = det.detect(img1)

kp0, des0 = det.detector.detectAndCompute(img0, None)  # left
kp1, des1 = det.detector.detectAndCompute(img1, None)  # right

kp0 = np.array(kp0);
kp1 = np.array(kp1)

# BFMatcher with default params
bf = cv2.BFMatcher(normType=cv2.NORM_L2)
matches = bf.knnMatch(des1, des0, k=2)  # img0 is train, img1 is query. Looking for right features in left

# Apply ratio test to select good features
good = []
for m, n in matches:
    if m.distance < LOWE_DISTANCE_RATIO * n.distance:
        good.append([m])

# getting the (u,v) points for the good matches
match_idxs = np.zeros((len(good), 2), dtype=int)
for i, m in enumerate(good):
    match_idxs[i] = [m[0].trainIdx, m[0].queryIdx]

# -----------------------------------------------------------------------------------------------------------------
#       Stereo Depth
# -----------------------------------------------------------------------------------------------------------------
print("Setting up stereo")

K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(data.calib.P_rect_10)
t = t[:3] / t[3]  # normalising translation vector
t.squeeze()

# Getting disparity:
left_uv = keyPoint_to_UV(kp0[match_idxs[:, 0]])
right_uv = keyPoint_to_UV(kp1[match_idxs[:, 1]])
d = np.linalg.norm(left_uv - right_uv, axis=1)
depth = disparity_to_depth(d, K, t)

ft_uvd = np.hstack((left_uv, depth.reshape((-1, 1))))
ft_uvd = ft_uvd[ft_uvd[:, 2] < MAX_DISTANCE]  # Validating points

# Building mesh
d_mesh = Delaunay(ft_uvd[:, :2])

δ = cv2.Laplacian(img0, cv2.CV_64F)
δ = np.abs(δ)

idx = np.argpartition(δ.flatten(), -INTERPOLATING_POINTS)[-INTERPOLATING_POINTS:]
gradient_pts = np.unravel_index(idx, img0.shape)

interpolated_uv = np.stack((gradient_pts[1], gradient_pts[0]), axis=-1)
# mesh_pts_3d = ft_uvd.copy()

interpolated_pts = barycentric_interpolation(d_mesh, ft_uvd, interpolated_uv)



# -----------------------------------------------------------------------------------------------------------------
#       Cost Calculation
# -----------------------------------------------------------------------------------------------------------------
print("Calculating costs")

new_census = np.abs(get_disparity_census_cost(interpolated_pts, img0, img1, K, t, num_disparity_levels=5))

# MAX_COST = 0.8  # Hardcoded values
# MIN_COST = 0.2  # Hardcoded values

MAX_COST = new_census[:, new_census.shape[1] // 2].mean()  # Using shifting boundaries
MIN_COST = new_census.min(axis=1).mean() + new_census.min(axis=1).std()  # Statistically defined minimum
FT_COSTS = np.abs(get_disparity_census_cost(ft_uvd, img0, img1, K, t, num_disparity_levels=1)).mean()
print("-----------------------------")
print("Cost Calculation:")

print(f"Mean {new_census.mean()}")
print(f"Max {new_census.max()}")
print(f"Min {new_census.min()}")

print(f"Setting MAX_COST to {MAX_COST}")
print(f"Setting MIN_COST to {MIN_COST}")
print(f"Average cost from features for mesh construction was {FT_COSTS}")

min_cost_idx = new_census.min(axis=1) < MIN_COST

# Using the best match in the window at this point. It might not be right
max_cost_idx = new_census.min(axis=1) > MAX_COST

# C_g = np.nan*np.ones((len(new_census), 4)) # (u,v,d,c)
C_g = -1 * np.ones((len(new_census), 4))  # (u,v,d,c)
C_b = -1 * np.ones((len(new_census), 4))  # (u,v,d, c)

# good points are those with the lowest cost
C_g[min_cost_idx] = np.hstack((interpolated_pts[min_cost_idx, :2],
                               depth_to_disparity(interpolated_pts[min_cost_idx, 2], K, t).reshape((-1, 1)),
                               new_census.min(axis=1)[min_cost_idx].reshape((-1, 1))))
C_g_d_arg = np.argmin(new_census, axis=1)
C_b_d_arg = np.argmax(new_census, axis=1)

# # Using the best match (new_census.min)
# C_b[max_cost_idx] = np.hstack((interpolated_pts[max_cost_idx, :2],
#                                depth_to_disparity(interpolated_pts[max_cost_idx, 2], K, t).reshape((-1, 1)),
#                                new_census.min(axis=1)[max_cost_idx].reshape((-1, 1))))

# Choosing bad points as compliment of good points
max_cost_idx = ~min_cost_idx
C_b[max_cost_idx] = np.hstack((interpolated_pts[max_cost_idx, :2],
                               depth_to_disparity(interpolated_pts[max_cost_idx, 2], K, t).reshape((-1, 1)),
                               new_census.min(axis=1)[max_cost_idx].reshape((-1, 1))))



print(f"Total num census cost pts {len(new_census)}")
print("Number of good interpolated points:", np.sum(np.sum(C_g, axis=1)!=-C_g.shape[1]))
print("Number of bad interpolated points:", np.sum(np.sum(C_b, axis=1)!=-C_b.shape[1]))


# -----------------------------------------------------------------------------------------------------------------
#       Resampling
# -----------------------------------------------------------------------------------------------------------------
print("Support Resampling")
pts_to_resample = C_b[np.all(C_b!=-1, axis=1),:3]

# Inplementing selection of high cost points for the occupancy grids

support_pts = support_resampling(img0, img1, pts_to_resample)
support_pts = support_pts[np.logical_and(np.logical_and(
    np.all(support_pts>0, axis=1), (support_pts[:,0]<img0.shape[1])), (support_pts[:,1]<img0.shape[0] ))]



# -----------------------------------------------------------------------------------------------------------------
#       Adjusting depth for resampled points
# -----------------------------------------------------------------------------------------------------------------
print("Adjusting depth for resampled points")



# Epipolar search
F, mask = cv2.findFundamentalMat(left_uv.astype(np.int32),right_uv.astype(np.int32),cv2.FM_LMEDS) # getting the fundamental matrix

# print("Found F:\n",F)
# Probably need to filter the matches used for this to only good points

# Searching for points along epipolar lines:
resampled = np.stack((pts_to_resample[:,0], pts_to_resample[:,1], get_depth_with_epipolar(img0, img1, pts_to_resample, K=K, t=t, F=F)), axis=-1)
resampled = resampled[resampled[:,2]<=MAX_DISTANCE]

support_resampled = np.stack((support_pts[:,0], support_pts[:,1], get_depth_with_epipolar(img0, img1, support_pts, K=K, t=t, F=F)), axis=-1)
support_resampled = support_resampled[support_resampled[:,2]<=MAX_DISTANCE]


# good_resampled = np.stack((C_g[:,0], C_g[:,1], get_depth_with_epipolar(img0, img1, C_g)), axis=-1)
# good_resampled = good_resampled[good_resampled[:,2]<=MAX_DISTANCE





new_pts = np.vstack((ft_uvd, resampled, support_resampled))
old_pts = np.vstack((ft_uvd, pts_to_resample, support_pts))




print(f"Now have {len(new_pts)} new interpolated points")
# Displaying output:

fig,ax = plt.subplots(2,2, figsize=(30,10))
fig.tight_layout(pad=3.0)

ax[0,0].imshow(img0, 'gray')
ax[0,0].axis('off')
plot_mesh(d_mesh, ft_uvd, a=ax[0,0])
sc = ax[0,0].scatter(ft_uvd[:,0], ft_uvd[:,1], c= ft_uvd[:,2], cmap='jet')
ax[0,0].set_title("Original mesh points")
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(sc, cax = cax)
cbar.set_label("Depth [m]")

ax[0,1].imshow(img0, 'gray')
ax[0,1].axis('off')

ind = old_pts[:,2]<MAX_DISTANCE
sc = ax[0,1].scatter(old_pts[ind,0], old_pts[ind,1], c= old_pts[ind,2], cmap='jet')
ax[0,1].set_title("Original interpolated points")
divider = make_axes_locatable(ax[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(sc, cax = cax)
cbar.set_label("Depth [m]")



ax[1,0].imshow(img0, 'gray')
ax[1,0].axis('off')
sc = ax[1,0].scatter(new_pts[:,0], new_pts[:,1], c=new_pts[:,2], cmap='jet')
ax[1,0].set_title("After resampling, before bounding")
divider = make_axes_locatable(ax[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(sc, cax = cax)
cbar.set_label("Depth [m]")


new_pts = new_pts[new_pts[:,2]<MAX_DISTANCE]


ax[1,1].imshow(img0, 'gray')
ax[1,1].axis('off')
sc = ax[1,1].scatter(new_pts[:,0], new_pts[:,1], c=new_pts[:,2], cmap='jet')
ax[1,1].set_title("After resampling and discarding all further than 100m away")
divider = make_axes_locatable(ax[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(sc, cax = cax)
cbar.set_label("Depth [m]")

plt.show()


fig, ax = plt.subplots(1,1, figsize=(20,10))
fig.tight_layout()
ax.axis('off')
ax.imshow(img0, 'gray')
plot_mesh(Delaunay(new_pts[:,:2]), new_pts, a=ax)


sc = ax.scatter(new_pts[:,0], new_pts[:,1], c=new_pts[:,2], cmap='jet')
ax.set_title(f"Mesh produced using {len(new_pts)} points")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(sc, cax = cax)
cbar.set_label("Depth [m]")

plt.show()

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
