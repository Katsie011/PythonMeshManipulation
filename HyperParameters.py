# import pykitti
#
# print("Loading Kitti")
# base = r"/home/kats/Datasets/KITTI_cvlibs/"
# date = "2011_09_26"
# drive = "0001"
# data = pykitti.raw(base, date, drive)
# print("Loaded")

import sys

import reconstruction.HuskyCalib as HuskyCalib

if "aru_core_lib" not in sys.modules:
    # you may need a symbolic link to the build of aru_core library
    import reconstruction.aru_core_lib.aru_py_mesh as aru_py_mesh

CWD = "/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation"
config_path = "/home/kats/Code/PythonMeshManipulation/TunableReconstruction/mesh_depth.yaml"
depth_est = aru_py_mesh.PyDepth(config_path)


# K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(data.calib.P_rect_10)
# t = t[:3] / t[3]  # normalising translation vector
# t.squeeze()

NUM_INITIAL_FEATURES = 200
LOWE_DISTANCE_RATIO = 0.8
MAX_DISTANCE = 60  # meters
MAX_DISPARITY = 0.5
MAX_NUM_FEATURES_DETECT = 2000
MIN_DISTANCE = 2
TRIANGLE_SAMPLES_PER_PIX_SQUARED = 1 / 10 ** 2
# DETECTOR = 'ORB'
DETECTOR = 'SIFT'
INTERPOLATING_POINTS = 200
NUM_SUPPORT_PTS_PER_OCCUPANCY_GRID = 2
RESAMPLING_ITERATIONS = 3

IMAGE_SHAPE = (720, 1280, 3)
PREDICTION_SHAPE = (384, 640)
