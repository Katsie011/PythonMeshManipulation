import cv2
import os

import tqdm

import utilities.HuskyDataHandler as husky




data_path = "/media/kats/Katsoulis31/Datasets/Husky/extracted_data/old_zoo/"
out_path = "/media/kats/Katsoulis31/Datasets/Husky/Training Data"
dataset = husky.DatasetHandler(data_path)

old_size = dataset.img_shape
new_size = (256, 512)

out_path_left = os.path.join(out_path, "image_00")
out_path_right = os.path.join(out_path, "image_01")

if not os.path.exists(out_path_left):
    os.makedirs(out_path_left)
if not os.path.exists(out_path_right):
    os.makedirs(out_path_right)

print("Resizing left images")
for i in tqdm.trange(len(dataset.left_image_files)):
    l_name = dataset.left_image_files[i]
    l_full_path = os.path.join(dataset.left_image_path, l_name)
    l_full_img = cv2.imread(l_full_path)
    l_resized = cv2.resize(l_full_img, new_size[::-1])

    l_resize_path = os.path.join(out_path_left, l_name)
    success = cv2.imwrite(l_resize_path, l_resized, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not success:
        print(f"Imwrite failed on frame{i}: {l_name}")
        break

print("Resizing right images")
for i in tqdm.trange(len(dataset.right_image_files)):
    r_name = dataset.right_image_files[i]
    r_full_path = os.path.join(dataset.right_image_path, r_name)
    r_full_img = cv2.imread(r_full_path)
    r_resized = cv2.resize(r_full_img, new_size[::-1])

    r_resize_path = os.path.join(out_path_right, r_name)
    success = cv2.imwrite(r_resize_path, r_resized, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not success:
        print(f"Imwrite failed on frame{i}: {l_name}")
        break


