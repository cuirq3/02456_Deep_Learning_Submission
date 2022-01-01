import pandas as pd
import numpy as np
import os

root_dir = r'E:\DL_data\osfstorage-archive\HELMET_DATASET\annotation'
img_dir = r'E:\DL_data\osfstorage-archive\HELMET_DATASET\image'
new_img_dir = r'E:\DL_data\osfstorage-archive\HELMET_DATASET\images'
new_label_dir = r'E:\DL_data\osfstorage-archive\HELMET_DATASET\labels'
img_idx = 0
img_width = 1920
img_height = 1080

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.csv'):
            label_path = root_dir + '/' + file
            df = pd.read_csv(label_path)
            df.sort_values(by='frame_id', inplace=True)
            now_frame_id = -1
            now_txt_dir = ''
            img_path = ''
            for _ , row in df.iterrows():
                if(now_frame_id is not int(row[1])):
                    now_frame_id = row[1]
                    img_idx += 1
                    img_path = img_dir + '/' + file.split('.')[0] + '/' + str(now_frame_id).zfill(2) + '.jpg'
                    os.rename(img_path, new_img_dir + '/' + str(img_idx).zfill(2) + '.jpg')
                    now_txt_dir = new_label_dir + '/' + str(img_idx).zfill(2) + '.txt'
                f = open(now_txt_dir, 'a')
                x = int(row[2])
                y = int(row[3])
                w = int(row[4])
                h = int(row[5])
                x_center = (x + x + w) / 2 / img_width
                y_center = (y + y + h) / 2 / img_height
                width = w / img_width
                height = h / img_height
                f.write('0 ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')
                f.close()