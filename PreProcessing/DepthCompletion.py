


import numpy as np
import cv2


# sys.path.insert(0, '/home/kats/Documents/Repos/aru-core/build/lib')
# import aru_py_mesh
import sys
sys.path.insert(0,"/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/Modular files")
import DatasetHandler

Handler = DatasetHandler.Dataset_Handler_Depth_Data('/media/kats/DocumentData/Data/data_depth_selection/depth_selection/')

img = Handler.first_image_left
sparse = Handler.first_depth


cv2.imshow("img", img)
cv2.imshow("depth", sparse)

cv2.waitKey(0)
cv2.destroyAllWindows()

original = img.astype(float)/255
marked = sparse.copy().astype(float)/255
new = -1*np.ones(original.shape)

isColored = sparse > 0.01                                # isColored as colorIm

n, m = original.shape
image_size = original.size

indices_matrix = np.arange(image_size).reshape(n, m, order='F').copy()  # indices_matrix as indsM

wd = 1  # The radius of window around the pixel to assess
nr_of_px_in_wd = (2 * wd + 1) ** 2  # The number of pixels in the window around one pixel
max_nr = image_size * nr_of_px_in_wd  # Maximal size of pixels to assess for the hole image
# (for now include the full window also for the border pixels)

row_inds = np.zeros(max_nr, dtype=np.int64)
col_inds = np.zeros(max_nr, dtype=np.int64)
vals = np.zeros(max_nr)

# ----------------------------- Interation ----------------------------------- #

length = 0  # length as len
pixel_nr = 0  # pixel_nr as consts_len
# Nr of the current pixel == row index in sparse matrix

# iterate over pixels in the image
for j in range(m):
    for i in range(n):

        # If current pixel is not colored
        if (not isColored[i, j]):
            window_index = 0  # window_index as tlen
            window_vals = np.zeros(nr_of_px_in_wd)  # window_vals as gvals

            # Then iterate over pixels in the window around [i,j]
            for ii in range(max(0, i - wd), min(i + wd + 1, n)):
                for jj in range(max(0, j - wd), min(j + wd + 1, m)):

                    # Only if current pixel is not [i,j]
                    if (ii != i or jj != j):  # TODO This should be an AND
                        row_inds[length] = pixel_nr
                        col_inds[length] = indices_matrix[ii, jj]
                        window_vals[window_index] = YUV[ii, jj, 0]
                        length += 1
                        window_index += 1

            center = YUV[i, j, 0].copy()  # t_val as center
            window_vals[window_index] = center
            # calculate variance of the intensities in a window around pixel [i,j]
            variance = np.mean(
                (window_vals[0:window_index + 1] - np.mean(window_vals[0:window_index + 1])) ** 2)  # variance as c_var
            # TODO replace variance calculation with np eq.
            sigma = variance * 0.6  # csig as sigma

            # Indeed, magic
            mgv = min((window_vals[0:window_index + 1] - center) ** 2)
            if (sigma < (-mgv / np.log(0.01))):
                sigma = -mgv / np.log(0.01)
            if (sigma < 0.000002):  # avoid dividing by 0
                sigma = 0.000002

            window_vals[0:window_index] = np.exp(
                -((window_vals[0:window_index] - center) ** 2) / sigma)  # use weighting funtion (2)
            window_vals[0:window_index] = window_vals[0:window_index] / np.sum(
                window_vals[0:window_index])  # make the weighting function sum up to 1
            vals[length - window_index:length] = -window_vals[0:window_index]

        # END IF NOT COLORED

        # Add the values for the current pixel
        row_inds[length] = pixel_nr

        col_inds[length] = indices_matrix[i, j]
        vals[length] = 1
        length += 1
        pixel_nr += 1

    # END OF FOR i
# END OF FOR j


# ---------------------------------------------------------------------------- #
# ------------------------ After Iteration Process --------------------------- #
# ---------------------------------------------------------------------------- #

# Trim to variables to the length that does not include overflow from the edges
vals = vals[0:length]
col_inds = col_inds[0:length]
row_inds = row_inds[0:length]




