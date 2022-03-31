import numpy as np
import cv2
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt

im_fig_size = [10, 7]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def render_points(points, img_shape=[375, 1242], normalise=False, dilation_size=1, colour_map=-1):
    valid = np.where(
        (points[:, 0] >= 0) & (points[:, 1] >= 0) & (points[:, 0] < img_shape[1]) & (points[:, 1] < img_shape[0]))
    points = points[valid].round().astype(int)
    img_shape = np.array(img_shape).reshape((-1))
    #     img_shape[:2] += 1

    if colour_map != -1:
        #         print("Using cmap")
        if len(img_shape) < 3:
            img_shape = np.hstack((img_shape, 3))
        rendered = np.zeros(img_shape, dtype=np.uint8)

        minima = min(points[:, 2])
        maxima = max(points[:, 2])

        norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=colour_map)
        cs = mapper.to_rgba(points[:, 2])[:, :3] * 255

        rendered[points[:, 1], points[:, 0]] = cs
        rendered = cv2.dilate(rendered, np.ones((dilation_size, dilation_size)))
        return rendered


    else:
        if len(img_shape) > 2:
            img_shape = img_shape[:2]

        rendered = np.zeros(np.array(img_shape) + 1)
        rendered[points[:, 1], points[:, 0]] = points[:, 2]
        if dilation_size > 1:
            rendered = cv2.dilate(rendered, np.ones((dilation_size, dilation_size)))
        return rendered


def pointcloud_to_img_frame(pointcloud, img_height, img_width, Tr, P, rtrn_reflectance=True, rtrn_depth=False):
    #     Trimming pointcloud for speedup
    pointcloud = pointcloud[pointcloud[:, 0] > 0].copy()  # Everything in front of camera

    xyz = pointcloud[:, :3]
    reflectance = pointcloud[:, 3]

    #     making the pointcloud homogenous
    xyz = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    #     xyz now [ X, Y, Z, 1]

    #     transforming lidar into camera frame
    cam_xyz = Tr.dot(xyz.T).T
    #     lidar_depth = cam_xyz[:, 2].copy()  #Everything in front of camera
    cam_xyz = cam_xyz / cam_xyz[:, 2].reshape((-1, 1))

    img_xyz = P.dot(cam_xyz.T).T  # image frame
    pixel_coords = img_xyz.round(0).astype(int)
    indicies = np.where(
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 0] < img_width)
        & (pixel_coords[:, 1] < img_height)
    )

    if rtrn_reflectance:
        rtrn = np.hstack((xyz[indicies, :-1].reshape((-1, xyz.shape[-1] - 1)), reflectance[indicies].reshape((-1, 1))))
    elif rtrn_depth:
        # pythag to get distance of ray to point.
        d = np.sqrt(np.sum(xyz[indicies][:, :2] ** 2, axis=1))
        rtrn = np.hstack((xyz[indicies, :-1].reshape((-1, xyz.shape[-1] - 1)), d.reshape((-1, 1))))
    else:
        rtrn = xyz[indicies]
    return rtrn


def pointcloud_to_image(pointcloud, img_height, img_width, Tr, P, useDepth=True):
    #     Trimming pointcloud for speedup
    xyzc = pointcloud_to_img_frame(
        pointcloud, img_height, img_width, Tr, P, rtrn_reflectance=(not useDepth), rtrn_depth=useDepth
    )
    xyz = xyzc[:, :-1]
    c = xyzc[:, -1]

    #     making the pointcloud homogenous
    xyz = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    #     xyz now [ X, Y, Z, 1]

    #     transforming lidar into camera frame
    cam_xyz = Tr.dot(xyz.T).T
    lidar_depth = cam_xyz[:, 2].copy()  # Everything in front of camera
    cam_xyz = cam_xyz / cam_xyz[:, 2].reshape((-1, 1))

    img_xyz = P.dot(cam_xyz.T).T

    pixel_coords = img_xyz.round(0).astype(int)
    indicies = np.where(
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 0] < img_width)
        & (pixel_coords[:, 1] < img_height)
    )

    pixel_coords = pixel_coords[indicies]
    pixel_coords.max(axis=0)
    render = np.zeros((img_height, img_width))

    for j, (u, v) in enumerate(pixel_coords[:, :2]):
        #         if useDepth: render[v, u] = lidar_depth[j]
        #         else:
        #             render[v, u] = (reflectance[j] * 255).astype(int)
        render[v, u] = c[j]

    return render


def plot_pointcloud_on_image(
        im, render, rtrn_img=False, dilate_kernel_size=2, grayscale=False, colormap=cv2.COLORMAP_HSV
):
    im = np.array(im).copy()
    fig_l, ax_l = plt.subplots(1, 1, figsize=im_fig_size)
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dst = (255 * (render.copy()) / render.max()).astype(np.uint8)
    dst = cv2.dilate(dst, kernel, iterations=1)
    inds = np.where(dst > 0)
    dst = cv2.applyColorMap(dst, colormap)

    if grayscale:
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        # ensure three image channels if there is not three.
        if len(im.shape) != 3:
            print("Converting to 3 channel image")
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    img_dist = im.copy()
    img_dist[inds] = dst[inds]
    if rtrn_img:
        return img_dist
    # ax_l[1].imshow(cv2.dilate(render_reflectance.copy(), kernel, iterations=1))
    # ax_l[1].imshow(img_ref)
    ax_l.imshow(img_dist)
    ax_l.axis("off")
    ax_l.set_title("Lidar plot coloured by depth", fontsize=im_fig_size[0])


def get_closest_triangles(samples, tri_obj, verticies=False, return_bcs=True, fix_tris=True, precision=7, debug=False):
    # find the simplicies for all contained points.
    tris = tri_obj.find_simplex(samples[:, :2])

    centroids = tri_obj.points[tri_obj.simplices].sum(axis=1) / 3

    inds = np.array(np.where(tris < 0)[0])

    if return_bcs:
        bcs = -99 * np.ones((samples.shape[0], 3))
        for i, t in enumerate(tris):
            pt = samples[i, :2]
            b = tri_obj.transform[t, :2].dot(np.transpose(pt.reshape((1, -1)) - tri_obj.transform[t, 2]))
            bcs[i] = np.c_[np.transpose(b), 1 - b.sum(axis=0)].round(precision)

        inds = np.array(np.where((bcs.round(precision).min(axis=1) < 0) + (bcs.max(axis=1).round(precision) > 1))[0])

        if debug:
            print("Number of initial wrong points:", inds.shape, np.sum(tris < 0))

    for i, pt in enumerate(samples[inds][:, :2]):
        dist = np.sum((centroids - pt) ** 2, axis=1)
        #         dist = np.sum((centroids-pt)**2, axis=1)
        tris[inds[i]] = dist.argmin()  # closest simplice

        if return_bcs:
            # the bc is wrong
            #                 if debug:print("Wrong BC: ", pt)
            #                 if debug:print("   BC:", b.round(2))
            #                 if debug:print("   Tri:", tri_obj.points[tri_obj.simplices[tris[inds[i]]]])

            if not fix_tris:
                bcs[inds[i]] = -np.ones((1, 3))
            else:
                verts = tri_obj.points[tri_obj.simplices[tris[inds[i]]]]
                d = np.sum((verts - pt) ** 2, axis=1)
                v_arg = d.argsort()
                #             get lines for all edges
                #             Get line from pt to centroid
                #             check where intersect is between verticies
                #             pt of intersect is closest.

                #             r = p2-p
                #             s = q2-q
                #             t and u are parameters
                #             line1 = p + t*r
                #             line2 = q + u*s
                #             Want: p + t*r = q + u*s
                #             t = (q − p) × s / (r × s)
                #             u = (q − p) × r / (r × s)
                #             if r × s ≠ 0 and 0 ≤ t ≤ 1 and 0 ≤ u ≤ 1
                #                 Then intersection between points
                p = pt
                r = centroids[tris[inds[i]]] - pt
                q = verts[v_arg[0]]  # Closest point

                edge_pt = q.copy()
                for v in verts[v_arg[1:]]:
                    s = v - q
                    c = np.cross(r, s)
                    if c == 0:
                        if debug: print("Divide by 0 in get_closest_triangles")
                        continue
                    t = np.cross((q - p), s) / c
                    u = np.cross((q - p), r) / c

                    if (0 <= t <= 1) and (0 <= u <= 1):
                        edge_pt = p + t * r
                        break
                b = tri_obj.transform[tris[inds[i]], :2].dot(
                    np.transpose(edge_pt.reshape((1, -1)) - tri_obj.transform[tris[inds[i]], 2])
                )
                bcs[inds[i]] = np.c_[np.transpose(b), 1 - b.sum(axis=0)]

    if verticies:
        cont_tri_vert = samples[tri_obj.simplices[tris]]
        return cont_tri_vert

    if return_bcs:
        bcs = bcs.round(precision)
        #         if debug:print("Other BCs:")
        if debug:
            print(
                "Number of wrong fixed points:",
                np.sum((bcs[inds].round(precision).min(axis=1) < 0) + (bcs[inds].max(axis=1).round(5) > 1)),
            )
        if debug:
            print("Number of wrong points:", np.sum(bcs.min(axis=1) < 0 + (bcs.max(axis=1).round(precision) > 1)))
        #         if bcs[i].min().round(2) < 0:
        #             if debug:print("Wrong BC 2nd round: \n", pt)
        #             if debug:print("   BC:",bcs[i].round(2))
        #             if debug:print(t)
        return tris, bcs

    return tris


def get_general_barycentric(tri, pts):
    bcs = np.zeros((pts.shape[0], 3))
    for i in range(pts.shape[0]):
        pt = pts[i, :2]
        t = tri.find_simplex(pt)
        b = tri.transform[t, :2].dot(np.transpose(pt.reshape((1, -1)) - tri.transform[t, 2]))
        bcs[i] = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
    return bcs


def get_mesh_error(
        mesh_obj, samples, mesh_points, error_ord=2, precision=5, bounding_shape=[], Fix_mesh=True, debug=True
):
    r"""
    Returns error from point to barycentric projection of point in triangle.
    If point not in any triangle, use barycentric defined point on closest edge of closest triangle
    
    Params:
        error_ord: ord for np.linalg.norm
            mean squared error if error_ord = 2
    
    
    Returns:
        ndarray (n, 3)
            - n points, Error in X, Y and Z
            Should be no X & Y error if in a triangle.
    """
    proj_samples = interpolate_mesh_pts(mesh_obj, samples, mesh_points, Fix_pts_outside=Fix_mesh)

    dist = samples - proj_samples

    if np.shape(bounding_shape)[0] > 0:
        valid = np.where(
            (proj_samples[:, 0] >= 0)
            & (proj_samples[:, 1] >= 0)
            & (proj_samples[:, 0] < bounding_shape[1])
            & (proj_samples[:, 1] < bounding_shape[0])
        )
        if debug:
            if dist.shape[0] - np.shape(valid)[-1]:
                print(f"{dist.shape[0] - np.shape(valid)[-1]} points found outside mesh")
                print(".... Disgarding pts outside")

        dist = dist[valid]

    if error_ord > 2:
        error = np.linalg.norm(dist, ord=error_ord) / dist.shape[0]
    else:
        error = (dist ** 2).sum() / dist.shape[0]

    #     Return Error.
    return error


def interpolate_mesh_pts(mesh_obj, samples, mesh_points, precision=6, Fix_pts_outside=True):
    r"""
    Assumes mesh is in xy plane

    Returns 3D interpolated points (or points on closest triangle edge if outside of mesh.)
    """
    #     Get closest triangles
    tris, bcs = get_closest_triangles(samples, mesh_obj, return_bcs=True, fix_tris=Fix_pts_outside)
    bcs = bcs.round(precision)
    #         Redundancy check to make sure points are on triangle edge for those outside mesh_obj

    #     Calculate Euclidean error in X, Y and Z
    pts = mesh_points[mesh_obj.simplices[tris]]
    return (bcs.reshape(-1, 3, 1) * pts).sum(axis=1)
