import os
import cv2
import numpy as np
import ModularFiles.HuskyDataHandler as husky
import shutil
import tqdm


def make_train_test_lists(data_dir, test_dir, training_dir, make_test=True, make_train=True, train_test_ratio=10,
                          make_time_files=False):
    print("Folders found:", os.listdir(data_dir))
    dir, folders, files = next(os.walk(data_dir))

    c = 0
    # I have no cooking clue why but as soon as you check or make folders this all breaks and you can't access the folders.
    """
    if not os.path.exists(os.path.join(test_dir, 'image_00/data')):
        os.makedirs(os.path.join(test_dir, 'image_00/data'))
    if not os.path.exists(os.path.join(test_dir, 'image_01/data')):
        os.makedirs(os.path.join(test_dir, 'image_01/data'))
    if not os.path.exists(os.path.join(test_dir, 'velodyne_points/data')):
        os.makedirs(os.path.join(test_dir, 'velodyne_points/data'))

    if not os.path.exists(os.path.join(training_dir, 'image_00/data')):
        os.makedirs(os.path.join(training_dir, 'image_00/data'))
    if not os.path.exists(os.path.join(training_dir, 'image_01/data')):
        os.makedirs(os.path.join(training_dir, 'image_01/data'))
    if not os.path.exists(os.path.join(training_dir, 'velodyne_points/data')):
        os.makedirs(os.path.join(training_dir, 'velodyne_points/data'))
    """

    if make_time_files:
        test_file = open(os.path.join(test_dir, "test_file_list.txt"), 'w')
        train_file = open(os.path.join(training_dir, 'training_file_list.txt'), 'w')

    for f in folders:
        husky_data = husky.Dataset_Handler(data_path=os.path.join(data_dir, f))
        print(f"Number of time synced frames is: {husky_data.num_frames}")
        flag_train = True
        flag_test = True
        file_list = zip(husky_data.left_image_files, husky_data.right_image_files)
        velo_flag = len(husky_data.lidar_files) > 0
        for i in tqdm.trange(len(husky_data.left_image_files)):
            (left, right) = next(file_list)

            if make_test:
                if i % train_test_ratio == 0:
                    im_left = np.array(husky_data.get_cam0(i, rectify=True))
                    im_right = np.array(husky_data.get_cam1(i, rectify=True))
                    if velo_flag:
                        velo = husky_data.get_lidar(i)

                    im_left_path = os.path.join(test_dir, 'image_00/data', left)
                    im_right_path = os.path.join(test_dir, 'image_01/data', right)

                    cv2.imwrite(im_left_path, im_left)
                    cv2.imwrite(im_right_path, im_right)
                    if velo_flag:
                        velo_path = os.path.join(test_dir, 'velodyne_points/data', husky_data.lidar_files[i])
                        np.save(velo_path, velo)


                    test_line = f"image_00/data/{left}\n"

                    if flag_test:
                        print(f"Test File Format:\n\t{test_line}", f"\n\timage_01/data{right}",
                              f'velodyne_points/data{husky_data.lidar_files[i]}')
                        flag_test = False
                    # if make_time_files:
                    #     test_file.write(test_line)

            if make_train:
                im_left = np.array(husky_data.get_cam0(i))
                im_right = np.array(husky_data.get_cam1(i))
                if velo_flag:
                    velo = husky_data.get_lidar(i)
                # continue
                im_left_path = os.path.join(training_dir, 'image_00/data', left)
                cv2.imwrite(im_left_path, im_left)

                im_right_path = os.path.join(training_dir, 'image_01/data', right)
                cv2.imwrite(im_right_path, im_right)

                train_line = f"image_00/data/{left} image_01/data/{right}\n"
                if make_time_files:
                    train_file.write(train_line)

                if velo_flag:
                    velo_path = os.path.join(training_dir, 'velodyne_points/data', husky_data.lidar_files[i])
                    np.save(velo_path, velo)
                if flag_train:
                    print(f"Train File Format:\n{train_line}")
                    flag_train = False

            c += 1

    print(f"Wrote out {c} files")


if __name__ == '__main__':
    data_dir = r"/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route C"
    training_dir = "/media/kats/Katsoulis3/Datasets/Husky/Training Data/Train_Route_C"
    test_dir = "/media/kats/Katsoulis3/Datasets/Husky/Testing Data/Test_Route_C"

    train_test_ratio = 10

    make_train_test_lists(data_dir=data_dir, test_dir=test_dir, training_dir=training_dir, make_train=False,
                          train_test_ratio=train_test_ratio, make_time_files=False)
