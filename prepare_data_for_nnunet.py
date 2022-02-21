import os
import shutil
from os.path import join

# folder where you saved your cropped data
data_original_dir = "G:/data_end/cropped_train/"

target_dir = "G:/data/data_endo_split/"

instruments = os.listdir(data_original_dir)

# generate folder to save data
folder_type = ['training','testing','validation']
split_type = ['images','labels']

# makedirs
for _folder in folder_type:
    target_path = target_dir + '/' + _folder
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for _run_type in split_type:
        target_path_sub = target_dir + '/' + _folder + '/' + _run_type
        if not os.path.exists(target_path_sub):
            os.mkdir(target_path_sub)

cnt = 0
cnt_test = 0
cnt_val = 0
for i in range(len(instruments)):
    if not i in [0, 2]:
        # for training
        case_dir = join(data_original_dir, instruments[i])
        case_files = os.listdir(join(case_dir, 'images'))
        for case_file in case_files:
            src_img = join(case_dir, 'images', case_file)
            src_label = join(case_dir, 'parts_masks', case_file.replace('jpg', 'png'))
            target_name = 'imgs_' + str(cnt) + '.png'
            target_name_image = 'imgs_' + str(cnt) + '.jpg'
            target_img = join(target_dir, 'training', 'images', target_name_image)
            target_label = join(target_dir, 'training', 'labels', target_name)
            shutil.copy(src_img, target_img)
            shutil.copy(src_label, target_label)
            cnt += 1
            print(src_img, src_label, target_img, target_label)
    elif i == 2:
        # for testing
        case_dir = join(data_original_dir, instruments[i])
        case_files = os.listdir(join(case_dir, 'images'))
        for case_file in case_files:
            src_img = join(case_dir, 'images', case_file)
            src_label = join(case_dir, 'parts_masks', case_file.replace('jpg', 'png'))

            target_name = 'imgs_' + str(cnt_test) + '.png'
            target_name_image = 'imgs_' + str(cnt_test) + '.jpg'
            target_img = join(target_dir, 'testing', 'images', target_name_image)
            target_label = join(target_dir, 'testing', 'labels', target_name)
            shutil.copy(src_img, target_img)
            shutil.copy(src_label, target_label)
            cnt_test += 1
            print(src_img, src_label, target_img, target_label)
    else:
        # for validation
        case_dir = join(data_original_dir, instruments[i])
        case_files = os.listdir(join(case_dir, 'images'))
        for case_file in case_files:
            src_img = join(case_dir, 'images', case_file)
            src_label = join(case_dir, 'parts_masks', case_file.replace('jpg', 'png'))

            target_name = 'imgs_' + str(cnt_val) + '.png'
            target_name_image = 'imgs_' + str(cnt_val) + '.jpg'
            target_img = join(target_dir, 'validation', 'images', target_name_image)
            target_label = join(target_dir, 'validation', 'labels', target_name)
            shutil.copy(src_img, target_img)
            shutil.copy(src_label, target_label)
            cnt_val += 1
            print(src_img, src_label, target_img, target_label)
