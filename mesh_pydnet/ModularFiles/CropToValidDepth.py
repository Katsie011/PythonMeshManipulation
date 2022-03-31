"""
Script to crop out the space in the image where there is no valid depth data
"""

import cv2
import sys
import os

sys.path.insert(0, '/home/kats/Documents/Repos/aru-core/build/lib')
import aru_py_mesh

sys.path.append('./ModularFiles/')
import DatasetHandler
import numpy as np


# noinspection PyPep8Naming
def get_interior_rectangle(depth_mask, plot=False, verbose=False):
    img2, contours, hierarchy = cv2.findContours(depth_mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    exteriorMask = ~depth_mask.astype(bool)
    contourPts = np.array(contours).copy().reshape(-1, 2)
    xmax, ymax = contourPts.max(axis=0)
    xmin, ymin = contourPts.min(axis=0)

    max_its = ymax - ymin
    for i in range(max_its):

        upper = [np.arange(xmin, xmax, dtype=int), np.ones(xmax - xmin, dtype=int) * ymin]
        lower = [np.arange(xmin, xmax, dtype=int), np.ones(xmax - xmin, dtype=int) * ymax]
        left = [np.ones(ymax - ymin, dtype=int) * xmin, np.arange(ymin, ymax, dtype=int)]
        right = [np.ones(ymax - ymin, dtype=int) * xmax, np.arange(ymin, ymax, dtype=int)]

        sum_upper = exteriorMask[upper[1], upper[0]].sum()
        sum_lower = exteriorMask[lower[1], lower[0]].sum()
        sum_left = exteriorMask[left[1], left[0]].sum()
        sum_right = exteriorMask[right[1], right[0]].sum()

        if sum_upper > 0:
            ymin += 1
        if sum_lower > 0:
            ymax -= 1
        if sum_left > 0:
            xmin += 1
        if sum_right > 0:
            xmax -= 1

        if (sum_upper + sum_left + sum_lower + sum_right) == 0:
            break

    # n_ymin = ymin
    # n_ymax = ymax
    # n_xmin = xmin
    # n_xmax = xmax

    for j in range(i):
        n_ymin = ymin-1; n_ymax = ymax+1; n_xmax = xmax+1; n_xmin = xmin-1;
        if n_xmin <0:
            n_xmin = 0
        if n_ymin<0:
            n_ymin = 0
        if n_xmax == exteriorMask.shape[1]:
            n_xmax = exteriorMask.shape[1]-1
        if n_ymax == exteriorMask.shape[0]:
            n_ymax = exteriorMask.shape[0]-1

        upper = [np.arange(n_xmin, n_xmax, dtype=int), np.ones(n_xmax - n_xmin, dtype=int) * n_ymin]
        lower = [np.arange(n_xmin, n_xmax, dtype=int), np.ones(n_xmax - n_xmin, dtype=int) * n_ymax]
        left = [np.ones(n_ymax - n_ymin, dtype=int) * n_xmin, np.arange(n_ymin, n_ymax, dtype=int)]
        right = [np.ones(n_ymax - n_ymin, dtype=int) * n_xmax, np.arange(n_ymin, n_ymax, dtype=int)]

        sum_upper = exteriorMask[upper[1], upper[0]].sum()
        sum_lower = exteriorMask[lower[1], lower[0]].sum()
        sum_left = exteriorMask[left[1], left[0]].sum()
        sum_right = exteriorMask[right[1], right[0]].sum()

        if sum_upper == 0:
            ymin = n_ymin
        if sum_lower == 0:
            ymax =n_ymax
        if sum_left == 0:
            xmin =n_xmax
        if sum_right == 0:
            xmax = n_xmax
    if (xmax - xmin) < 0:
        print("Image with zero width")
        xmin = 0
        xmax = exteriorMask.shape[1]-1
    if (ymax - ymin) < 0:
        print("Image with zero height")
        ymin = 0
        ymax = exteriorMask.shape[0]-1

    if verbose:
        print("Found the following Coordinates:", "\n________________\n",
              "ymin:", ymin, "\n",
              "ymax:", ymax, "\n",
              "xmin:", xmin, "\n",
              "xmax:", xmax, "\n")

    if plot:
        tmp = cv2.cvtColor(depth_mask.astype("uint8")*255, cv2.COLOR_GRAY2BGR)
        tmp = cv2.rectangle(tmp, (xmax, ymax),
                            (xmin, ymin), (13, 123, 34), 4)
        cv2.imshow("Resulting depth", tmp)
        cv2.waitKey(0)
        cv2.destroyWindow("Resulting depth")

    return [(xmax, ymax), (xmin, ymin)]

def crop_img(img, xmax, ymax, xmin, ymin):
    if len(np.shape(img))==3:
        new_img = img[ymin:ymax, xmin:xmax, :]
    else:
        new_img = img[ymin:ymax, xmin:xmax]
    return new_img

def reframe_dataset(dense_dir, img_dir):
    dense_filelist = []

    for root, dirs, files in os.walk(dense_dir):
        for file in files:
            dense_filelist.append(os.path.join(root, file))
    dense_filelist.sort()

    img_filelist = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            img_filelist.append(os.path.join(root, file))
    img_filelist.sort()

    depth_out_dir = os.path.join(dense_dir, "cropped")
    image_out_dir = os.path.join(img_dir, "cropped")

    data = {
        "image": [x for x in img_filelist if (x.endswith(".png"))],
        "depth": [x for x in dense_filelist if x.endswith("dense_depth.npy")],
    }
    if len(data["image"]) != len(data["depth"]):
        print("Error: incomplete file list. Lengths do not match")
        return 404

    for f_num in range(len(data["image"])):
        img_file = data["image"][f_num]
        depth_file = data["depth"][f_num]

        img_root, img_head = os.path.split(img_file)
        depth_root, depth_head = os.path.split(depth_file)

        img_path = os.path.join(image_out_dir, img_head)
        depth_path = os.path.join(depth_out_dir, depth_head)

        d = np.load(depth_file)
        im = cv2.imread(img_file)

        d_mask = d>0

        (xmax, ymax), (xmin, ymin) = get_interior_rectangle(d_mask)

        d_new = crop_img(d, xmax, ymax, xmin, ymin)
        im_new = crop_img(im, xmax, ymax, xmin, ymin)

        # cv2.imshow("d_new", d_new.astype("uint8"))
        # cv2.imshow("im_new", im_new.astype("uint8"))
        #
        # cv2.waitKey(0)

        # cv2.destroyAllWindows()

        cv2.imwrite(img_path, im_new)
        np.save(depth_path, d_new)
        print(f_num)
        print("Saved to ", img_path)
        print("Saved to ", depth_path)
        print()






if __name__ == "__main__":
    print("Hi")
    Handler = DatasetHandler.Dataset_Handler_Depth_Data(
        '/media/kats/DocumentData/Data/data_depth_selection/depth_selection/')
    depth_est = aru_py_mesh.PyDepth("/home/kats/Documents/Repos/aru-core/src/mesh/config/mesh_depth.yaml")
    cmap_gt = r"/home/kats/Documents/My Documents/Datasets/data_depth_selection/depth_selection/val_selection_cropped/dense_groundtruth"
    data_dir = r"/home/kats/Documents/My Documents/Datasets/data_depth_selection/depth_selection/val_selection_cropped"
    dense_dir = os.path.join(data_dir, "dense_groundtruth")
    mask_dir = os.path.join(data_dir, "depth_masks")
    img_dir = os.path.join(data_dir, "image")


    d = depth_est.create_dense_depth(Handler.first_lidar_img)[1].astype('uint8')
    # d_int = cv2.normalize(d, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    d_mask = d >-1
    # print(d_mask)

    # cv2.imshow("Depth", d)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    (xmax, ymax), (xmin, ymin) = get_interior_rectangle(depth_mask=d_mask)
    # cv2.imshow("cropped img", crop_img(Handler. first_image_left, xmax, ymax, xmin, ymin))
    # cv2.imshow("cropped depth", crop_img(d.astype("uint8"), xmax, ymax, xmin, ymin))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    reframe_dataset(dense_dir, img_dir)
