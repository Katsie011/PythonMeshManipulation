import numpy as np
import os
import tqdm

def cvt_all_in_dir(data_dir):
    r"""
    Converts all the .npy files found in the directory into .csv
    """

    file_list = os.listdir(data_dir)
    for f in tqdm.tqdm(file_list):
        fname, ext =os.path.splitext(f)
        if ext =='.npy':
            tmp = np.load(os.path.join(data_dir, f))
            np.savetxt(os.path.join(data_dir, fname + ".csv"), tmp, delimiter=',')


if __name__ == '__main__':
    data = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo_new/"

    for d in os.listdir(data):
        directory = os.path.join(data,d,'velodyne_points/data')
        print(d)
        cvt_all_in_dir(directory)

