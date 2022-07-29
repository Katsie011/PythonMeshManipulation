import os
import cv2
import numpy as np
import tqdm


def make_train_test_lists(training_dir):
    print("Folders found:", os.listdir(training_dir))
    # c = 0
    with open(os.path.join(training_dir, 'training_file_list.txt'), 'w') as train_file:
        left_files = os.listdir(os.path.join(training_dir, 'image_00'))
        right_files= os.listdir(os.path.join(training_dir, 'image_01'))
        left_files.sort()
        right_files.sort()
        flag_train=False
        for i in tqdm.trange(len(left_files)):
            left = left_files[i]
            right = right_files[i]
            train_line = f"image_00/{left} image_01/{right}\n"
            train_file.write(train_line)

            if flag_train:
                print(f"Train File Format:\n{train_line}")
                flag_train = False

            # c += 1

    print(f"Wrote out {i} files")


if __name__ == '__main__':
    # training_dir = r"/media/kats/Katsoulis3/Datasets/Husky/Training Data/old_zoo/"
    training_directory = r"/media/kats/Katsoulis31/Datasets/Husky/Training Data"

    make_train_test_lists(training_dir=training_directory)
