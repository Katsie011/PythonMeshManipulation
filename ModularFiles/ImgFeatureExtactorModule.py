"""
A little script to extract a series of points from an image.
creates a detector class

Tested and run on opencv 3.6.4

"""
# if ~('cv2' in dir()):
#     import cv2
#     # print("Imported cv2")
#
# if ~('numpy' in dir()):
#     import numpy as np
#     # print("Imported numpy as np")

import cv2
import numpy as np


class FeatureDetector:
    def __init__(self, det_type='ORB', max_num_ft=500):
        r"""
        Sets up a feature extractor given:
            - Type of detector needed.
                - orb, sift or surf.
            - Max number of keypoints
        """
        self.detector_type = det_type
        self.num_ft = max_num_ft
        # self.kp = []

        if det_type.lower() == 'sift':
            self.detector = cv2.SIFT_create(max_num_ft)
        elif det_type.lower() == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create(max_num_ft)
        # elif type.lower() == 'orb':
        else:
            self.detector = cv2.ORB_create(nfeatures=max_num_ft)

        self.bf = cv2.BFMatcher()

    def detect(self, image):
        """
        Detects image features and returns keypoints and descriptors
        """
        if type(image) == "NoneType":
            print("No image input")
            return -1
        if np.shape(image)[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(image, None)

    def kp_to_pts(self, kps):
        """
        Returns the keypoint's location in u,v
        """
        uv = np.zeros((len(kps), 2))

        for i, pt in enumerate(kps):
            u, v = pt.pt
            uv[i] = [u, v]

        return uv

    def get_matches(self, des1, des2, lowes_ratio=0.75):
        matches = self.bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < lowes_ratio * n.distance:
                good.append(m)

        good = sorted(good, key=lambda x: x.distance)

        return good


if __name__ == "__main__":
    img = cv2.imread("test_image.jpg")

    ft_det = FeatureDetector(det_type='SIFT', max_num_ft=500)
    print("SIFT detector returned", len(ft_det.detect(img)), "features")

    ft_det = FeatureDetector(det_type='SURF', max_num_ft=500)
    print("SURF detector returned", len(ft_det.detect(img)), "features")

    ft_det = FeatureDetector(det_type='ORB', max_num_ft=5000)
    print("ORB detector returned", len(ft_det.detect(img)), "features")

    print("Done")
