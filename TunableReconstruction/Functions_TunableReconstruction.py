# -----------------------------------------------------------------------------------------------------------------
#       Importing libraries
# -----------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
# from scipy.spatial import Delaunay
from scipy.spatial.distance import hamming
import numpy as np
from reconstruction.HyperParameters import *



# -----------------------------------------------------------------------------------------------------------------
#       Feature Extraction
# -----------------------------------------------------------------------------------------------------------------
def keyPoint_to_UV(kps):
    """ returns np array of len(kp)x2"""
    uv = np.zeros((len(kps), 2))
    for i, k in enumerate(kps):
        pt = k.pt
        uv[i] = pt

    return uv

def get_depth_pts(det, img, depth):
    u, v = keyPoint_to_UV(det.detect(img)).T
    u_d = np.round(u * depth.shape[1] / img.shape[1]).astype(int)
    v_d = np.round(v * depth.shape[0] / img.shape[0]).astype(int)
    d = depth[v_d, u_d]

    return np.stack((u, v, d), axis=-1)

# -----------------------------------------------------------------------------------------------------------------
#       Stereo Depth
# -----------------------------------------------------------------------------------------------------------------
def disparity_to_depth(d, K, t):
    # Assuming that fx will equal fy
    f = K[0, 0]
    if len(t.shape) > 1:
        t = t.squeeze()
    B = t[0]

    Z = B * f / d
    return Z


def depth_to_disparity(Z, K, t):
    # Assuming that fx will equal fy
    f = K[0, 0]
    if len(t.shape) > 1:
        t = t.squeeze()
    B = t[0]
    if np.size(Z)>1:
        Z[Z==0] = 0.000001
    else:
        if Z==0: Z=0.00001
    d = B * f / Z
    return d


# -----------------------------------------------------------------------------------------------------------------
#       Triangles
# -----------------------------------------------------------------------------------------------------------------

def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def get_convex_hull_mask(pts_2d, img):
    # 1. get a convex hull for the mesh
    hull = cv2.convexHull(pts_2d.astype(np.float32), False)
    hull = hull.reshape((-1, 2))

    # 2. turn into a mask
    mask = np.stack((np.zeros(img.shape, np.uint8),) * 3, axis=-1)
    mask = cv2.drawContours(mask, [hull.astype(int)], 0, (255, 255, 255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask


# mask = get_convex_hull_mask(ft_uvd[:,:2], img0)
# plt.imshow(mask, 'gray')
# plt.show()

def valid_bcs(b):
    return np.sign(b).sum(axis=1) == 3


def Barycentric(pts, triangle):
    r"""
    Compute barycentric coordinates (u, v, w) for
    point p with respect to triangle (a, b, c)

    Parameters: pts - The sample points that you want the barycentric coordinates for
                tri - The triangle containing the points

    Output: Barycentric coords in form [??1, ??2, ??3]
    """

    a, b, c = triangle[:, :2]
    if pts.shape[1] == 3:
        pts = pts[:, :2]

    v0 = b - a
    v1 = c - a
    v2 = pts - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array((u, v, w)).T


def get_barycentric_coords(mesh, pts):
    bcs = np.zeros((pts.shape[0], 3))
    for i in range(pts.shape[0]):
        pt = pts[i, :2]
        t = mesh.find_simplex(pt)
        b = mesh.transform[HuskyCalib.t_cam0_velo, :2].dot(np.transpose(pt.reshape((1, -1)) - mesh.transform[HuskyCalib.t_cam0_velo, 2]))
        bcs[i] = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
    return bcs


def barycentric_interpolation(mesh, mesh_pts_3d, sample_pts, valid_only=True):
    r"""
    Interpolates the given samples to get their depth values based on their position on a mesh
    Uses Barycentric coordinates to determine the depth at each (u,v) sample

    """

    interpolated = -1 * np.ones((sample_pts.shape[0], 3))
    for i in range(sample_pts.shape[0]):
        pt = sample_pts[i]
        tri = mesh.find_simplex(pt)
        b = mesh.transform[tri, :2].dot(np.transpose(pt.reshape((1, -1)) - mesh.transform[tri, 2]))
        bcs = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
        if valid_only:
            if not np.all(valid_bcs(bcs)):
                continue
        interpolated[i] = bcs @ mesh_pts_3d[mesh.simplices[tri]]

    if valid_only:
        interpolated = interpolated[np.all(interpolated != -1, axis=1), :]

    return interpolated


def interpolate_pts(imgl, d_mesh, ft_uvd, verbose=False):
    ?? = cv2.Laplacian(imgl, cv2.CV_64F)
    ?? = np.abs(??)

    idx = np.argpartition(??.flatten(), -INTERPOLATING_POINTS)[-INTERPOLATING_POINTS:]
    gradient_pts = np.unravel_index(idx, imgl.shape)
    interpolated_uv = np.stack((gradient_pts[1], gradient_pts[0]), axis=-1)
    interpolated_pts = barycentric_interpolation(d_mesh, ft_uvd, interpolated_uv)
    if verbose: print(f"Interpolated and returning {len(interpolated_pts)} points")
    return interpolated_pts


def plot_mesh(mesh, mesh_pts_3d, a=None, use_depth=True, cmap_=cm.jet, max_distance=MAX_DISTANCE):
    if isinstance(a, type(None)):
        print("No Axis input, creating new figure")
        f, a = plt.subplots(1, 1)
    tris = mesh_pts_3d[mesh.simplices]

    for i, verticies in enumerate(tris):
        verticies = tris[i]
        vert_plot = verticies[[0, 1, 2, 0], :]
        if use_depth:
            a.plot(vert_plot[:, 0], vert_plot[:, 1], color=cmap_((tris[i, :, 2].sum() / 3) / max_distance))
        else:
            a.plot(vert_plot[:, 0], vert_plot[:, 1], color=cmap_(i / len(mesh.simplices)))

        #         print(f"For point w depth {(tris[i, :,2].sum()/3)/MAX_DISTANCE}, color is {cm.jet((tris[i,:, 2].sum()/3)/MAX_DISTANCE)}")
        a.scatter(verticies[:, 0], verticies[:, 1])


# -----------------------------------------------------------------------------------------------------------------
#       Cost Functions
# -----------------------------------------------------------------------------------------------------------------

def census(window):
    mid_pixel = (np.array((window.shape)) / 2).round().astype(int)
    mid_pixel = window[mid_pixel[0], mid_pixel[1]]
    c = window > mid_pixel
    c = c.flatten().astype(int)
    return c


def get_census_cost(w1, w2):
    r"""Return census cost between two windows"""
    c = hamming(census(w1), census(w2))
    return c


def get_disparity_census_cost(vud, imgL, imgR, K, t, num_disparity_levels=3, window_size=5, verbose=False):
    # some checks to make sure that you can slice all the way to the left or right
    # will need to deal with edges of images

    pane = window_size // 2

    f = K[0, 0]
    if len(t.shape) > 1:
        t = t.squeeze()
    B = t[0]

    vu = vud[:, :2].round().astype(int)
    d = vud[:, 2]
    vu2 = vu.copy()

    vu2[:, 0] = vu[:, 0] - B * f / d  # shifting for disparity
    vu2 = vu2.round().astype(int)

    costs = -1 * np.ones((len(vud), num_disparity_levels))

    disparity_levels = np.arange(-num_disparity_levels // 2, num_disparity_levels // 2)

    #     if vud[0]<pane:
    #         raise Exception("vu of Census window too close to edge of image") # Just until this condition is dealt with

    if len(vud.shape) == 1:
        vud.reshape((1, -1))

    for i in range(len(vu)):
        for j in disparity_levels:

            w1 = imgL[vu[i, 1] - pane: vu[i, 1] + pane + 1, vu[i, 0] - pane:vu[i, 0] + pane + 1]
            if np.size(w1) < window_size ** 2:
                if verbose: print("Too close to the edge")
                #                 raise Exception("vu of Census window too close to edge of image") # Just until this condition is dealt with
                if verbose: print("w2 missing values")
                if verbose: print("w2:\n", w1)
                if verbose: print(f"i{i}, j{j}")
                if verbose: print("Slicing at:", (vu[i, 1] - pane, vu[i, 1] + pane + 1),
                                  (vu[i, 0] - pane, vu[i, 0] + pane + 12))
                if verbose: print("Img shape:", imgR.shape)
                continue
            #        Therefore: x' = x - Bf/Z

            u2 = (vu2[i, 0] - pane, vu2[i, 0] + pane + 1)
            v2 = (vu2[i, 1] - pane + j, vu2[i, 1] + pane + 1 + j)
            w2 = imgR[v2[0]:v2[1], u2[0]:u2[1]]
            if np.size(w2) < window_size ** 2:
                if verbose: print("Too close to the edge")
                #                 raise Exception("vu of Census window too close to edge of image") # Just until this condition is dealt with
                if verbose: print("w2 missing values")
                if verbose: print("w2:\n", w2)
                if verbose: print(f"i{i}, j{j}")
                if verbose: print("Slicing at:", v2, u2)
                if verbose: print("Img shape:", imgR.shape)
                continue
            costs[i, j] = hamming(census(w1), census(w2))  # census cost

    return costs


def epipolar_search(imgl, imgr, poi, epipolar_search_size, epipolar_line=None, window_size=5):
    r""" Epipolar search for points that correspond.

    Assuming: rectified stereo image.
    Will only look in u direction of image
    """

    w = window_size // 2
    #     us = np.arange(-epipolar_search_size//2, epipolar_search_size//2+1, dtype=int)+poi[0].round().astype(int)
    us = np.arange(-epipolar_search_size, 1, dtype=int) + poi[0].round().astype(
        int)  # searching the right image for a match so the point will lie on the left of where it is in the left img

    if np.all(epipolar_line != None):
        a, b, c = epipolar_line.T
        vs = np.round((-c - a * u) / b).astype(int).squeeze()
    else:
        vs = poi[1] * np.ones(len(us))

    target = imgl[poi[1].astype(int) - w:poi[1].astype(int) + w + 1, poi[0].astype(int) - w:poi[0].astype(int) + w + 1]

    #     plt.imshow(target, 'gray'); plt.title("target"); plt.show()

    scores = np.stack((-(us - poi[0]), -1 * np.ones(len(us))), axis=-1)
    for i, pt in enumerate(zip(us, vs.round().astype(int))):
        if w < pt[0] < imgl.shape[1] - w and w < pt[1] < imgl.shape[0] - w:
            search_area = imgr[pt[1] - w:pt[1] + w + 1, pt[0] - w:pt[0] + w + 1]
            if target.shape != search_area.shape:
                print(f"Target shape:{target.shape} \t Search shape:{search_area.shape}")
                continue
            scores[i, 1] = get_census_cost(target, search_area)
    #             plt.imshow(search_area, 'gray');plt.title(f"i:{i}, u:{u}");plt.show()

    return scores[::-1]


def get_depth_with_epipolar(imgl, imgr, pts, K, t, F, window_size=10, max_distance=MAX_DISTANCE,
                            min_distance=MIN_DISTANCE,
                            use_epipolar_line=False):
    e_s = depth_to_disparity(min_distance, K, t).round().astype(int)
    ds = np.zeros(len(pts))

    for j, pt in enumerate(pts):
        if use_epipolar_line:
            line = cv2.computeCorrespondEpilines(pt[:, 2], 1, F)
            c = epipolar_search(imgl, imgr, pt, e_s, epipolar_line=line, window_size=10)
        else:
            c = epipolar_search(imgl, imgr, pt, e_s, window_size=10)

        ds[j] = get_epipolar_cost_min(c)

    new_d = ds.copy()
    new_d[ds == 0] = max_distance
    new_d[ds != 0] = disparity_to_depth(ds[ds != 0], K, t)
    #     new_d[new_d>max_distance] = max_distance
    return new_d


def get_epipolar_cost_min(c):
    pos = 0
    c_min = c[0, 1]
    for i_, cost in c:
        if cost < c_min:
            c_min = cost
            pos = i_
    return pos


# -----------------------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------------------


def support_resampling(imgl, imgr, pts, sz_occ=32,  census_window_size=10,
                       pts_per_occ =  NUM_SUPPORT_PTS_PER_OCCUPANCY_GRID, verbose=False):
    to_sample = -np.ones((len(pts)*pts_per_occ, 3))
    c = 0
    w = census_window_size//2
    occ = -1 * np.ones((sz_occ, sz_occ))

    for i, [u, v, d] in  enumerate(pts.round().astype(int)):
        key = imgl[v - w : v+w+1, u-w : u+w+1]

        for x_i,x in enumerate(np.arange(u-sz_occ//2, u+sz_occ//2, dtype=int)):
                for y_i, y in enumerate(np.arange(v-sz_occ//2, v+sz_occ//2, dtype=int)):
                    l = x-w
                    r = x+w+1
                    up = y-w
                    down = y+w+1
                    check_window = imgr[up:down, l:r]
                    if key.shape != check_window.shape != (census_window_size, census_window_size):
                        if verbose:
                            print("Window sizes not equal")
                            print(check_window.shape, "!=", key.shape)
                        break
                    occ[y_i,x_i] = hamming(census(key), census(check_window))
                if key.shape != check_window.shape != (census_window_size, census_window_size):
                    break

        resample = np.unravel_index(np.argsort(occ.flatten())[-pts_per_occ:], occ.shape)
        #np.argpartition(??.flatten(), -pts_per_occ)[-pts_per_occ:]
        #resmpl = np.unravel_index(np.argsort(occ.flatten())[-NUM_SUPPORT_PTS_PER_OCCUPANCY:], occ.shape)
        r = np.stack((u - (resample[1]-sz_occ//2), v- (resample[0]-sz_occ//2), occ[resample]), axis=-1)
        to_sample[c:c+len(r)] =r
        c += len(resample)

    return to_sample






def calculate_costs(imgl, imgr, interpolated, ft_uvd, verbose=True):
    print("Calculating costs")
    new_census = np.abs(get_disparity_census_cost(interpolated, imgl, imgr, HuskyCalib.left_camera_matrix,HuskyCalib.t_cam0_velo, num_disparity_levels=5))

    # MAX_COST = 0.8  # Hardcoded values
    # MIN_COST = 0.2  # Hardcoded values

    MAX_COST = new_census[:, new_census.shape[1] // 2].mean() \
               - new_census[:, new_census.shape[1] // 2].std()  # Using shifting boundaries
    MIN_COST = new_census.min(axis=1).mean() - new_census.min(axis=1).std()  # Statistically defined minimum
    FT_COSTS = np.abs(get_disparity_census_cost(ft_uvd, imgl, imgl, HuskyCalib.left_camera_matrix,HuskyCalib.t_cam0_velo, num_disparity_levels=1)).mean()
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
                     depth_to_disparity(interpolated[min_cost_idx, 2], HuskyCalib.left_camera_matrix, t).reshape((-1, 1)),
                     new_census.min(axis=1)[min_cost_idx].reshape((-1, 1))))
    # c_g_d_arg = np.argmin(new_census, axis=1)
    # c_b_d_arg = np.argmax(new_census, axis=1)

    # # Using the best match (new_census.min)
    # c_b[max_cost_idx] = np.hstack((interpolated_pts[max_cost_idx, :2],
    #                                depth_to_disparity(interpolated_pts[max_cost_idx, 2], HuskyCalib.left_camera_matrix t).reshape((-1, 1)),
    #                                new_census.min(axis=1)[max_cost_idx].reshape((-1, 1))))

    # Choosing bad points as compliment of good points
    max_cost_idx = ~min_cost_idx
    c_b = np.hstack((interpolated[max_cost_idx, :2],
                     depth_to_disparity(interpolated[max_cost_idx, 2], HuskyCalib.left_camera_matrix, t).reshape((-1, 1)),
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
    new_d = get_depth_with_epipolar(imgl, imgr, support_pts, K=HuskyCalib.left_camera_matrix, t=HuskyCalib.t_cam0_velo, F=None, use_epipolar_line=False)
    resampled = np.stack((support_pts[:, 0], support_pts[:, 1], new_d), axis=-1)
    resampled = resampled[resampled[:, 2] <= MAX_DISTANCE]

    if eval_resampling_costs:
        c_bf = get_disparity_census_cost(support_pts, imgl, imgr, HuskyCalib.left_camera_matrix, HuskyCalib.t_cam0_velo, num_disparity_levels=1).mean()
        c_af = get_disparity_census_cost(resampled, imgl, imgr, HuskyCalib.left_camera_matrix, HuskyCalib.t_cam0_velo, num_disparity_levels=1).mean()
        if verbose: print(f"Resampled {len(support_pts)} support points")
        if verbose: print(f"Avg resampling cost before: {c_bf} \t after: {c_af}")

    return resampled
