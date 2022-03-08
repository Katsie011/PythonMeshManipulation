
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

    Output: Barycentric coords in form [ϕ1, ϕ2, ϕ3]
    """

    a, b, c = tri[:, :2]
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
        b = mesh.transform[t, :2].dot(np.transpose(pt.reshape((1, -1)) - mesh.transform[t, 2]))
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
        interpolated[i] = bcs @ mesh_pts_3d[d_mesh.simplices[tri]]

    if valid_only:
        interpolated = interpolated[np.all(interpolated != -1, axis=1), :]

    return interpolated


def plot_mesh(mesh, mesh_pts_3d, a=None, use_depth=True, cmap_=cm.jet):
    if isinstance(a, type(None)):
        print("No Axis input, creating new figure")
        f, a = plt.subplots(1, 1)
    tris = mesh_pts_3d[mesh.simplices]

    for i, verticies in enumerate(tris):
        verticies = tris[i]
        vert_plot = verticies[[0, 1, 2, 0], :]
        if use_depth:
            a.plot(vert_plot[:, 0], vert_plot[:, 1], color=cmap_((tris[i, :, 2].sum() / 3) / MAX_DISTANCE))
        else:
            a.plot(vert_plot[:, 0], vert_plot[:, 1], color=cmap_(i / len(mesh.simplices)))

        #         print(f"For point w depth {(tris[i, :,2].sum()/3)/MAX_DISTANCE}, color is {cm.jet((tris[i,:, 2].sum()/3)/MAX_DISTANCE)}")
        a.scatter(verticies[:, 0], verticies[:, 1])




# -----------------------------------------------------------------------------------------------------------------
#       Cost Functions
# -----------------------------------------------------------------------------------------------------------------

def census(window):
    mid_pixel = (np.array((window.shape))/2).round().astype(int)
    mid_pixel = window[mid_pixel[0], mid_pixel[1]]
    c = window>mid_pixel
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




# -----------------------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------------------

