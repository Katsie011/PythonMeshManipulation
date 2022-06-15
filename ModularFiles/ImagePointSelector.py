"""
A little script to manually select a series of points from an image.
Returns the coordinates of the points in pixels.


If a Z component is required, the dense depth image is sampled at the same points to get the depth.

Tested and run on opencv 4.5 but built for opencv 3.6.4

"""
if ~('cv2' in dir()):
    import cv2

    print("Imported cv2")

if ~('numpy' in dir()):
    import numpy as np

    print("Imported numpy as np")


# if ~('matplotlib.pyplot' in dir()):
#     import matplotlib.pyplot as plt
#
#     print("Imported matplotlib.pyplot as plt")


def image_point_selector(img, depth_map=None):
    r"""
    Returns selected points from an image.

    Return types:
        If no dense depth map: (nx2) ndarray
        Else: (nx3) ndarray
    """
    if type(img) == "NoneType":
        print("No image given")
        return -1
    if np.shape(img)[-1] ==3:
        image = img.copy()
    else:
        image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    have_pts = False
    w_name = "Press 'g' to generate mesh"
    pois = []
    refPt = []

    def click_points(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            pois.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 255, 0), 2)
            cv2.imshow(w_name, image)

    clone = img.copy()
    cv2.namedWindow(w_name)
    cv2.setMouseCallback(w_name, click_points)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow(w_name, image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("g"):
            break
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()

    pois = np.array(pois)
    if type(depth_map) == 'None':
        return pois

    p_depth = np.zeros((len(pois), 3))
    p_depth[:, :2] = pois
    p_depth[:, 2] = depth_map[pois[:, 1], pois[:, 0]]

    return p_depth


if __name__ == "__main__":
    img = cv2.imread("test_image.jpg")

    print(image_point_selector((img)))
