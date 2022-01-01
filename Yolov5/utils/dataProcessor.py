import pandas as pd
import shutil
import os

root_dir = r'E:\DL_data\HELMET\annotation'
# root_dir = r'E:\DL_data\HELMET\annotation_part'
img_dir = r'E:\DL_data\HELMET\image'
new_img_dir = r'E:\DL_data\HELMET\images'
new_label_dir = r'E:\DL_data\HELMET\labels'
new_helmet_label_dir = r'E:\DL_data\HELMET\helmet_labels'
img_idx = 0
img_width = 1920
img_height = 1080


def encodeHelmet(helmet_info):
    helmet_info_encoding = ['0', '0', '0', '0', '0', '0', '0', '0', '0']
    if 'D' in helmet_info:
        helmet_info_encoding[4] = '1'

    if 'P0' in helmet_info:
        helmet_info_encoding[0] = '1'
        helmet_info_encoding[5] = '1'
    if 'P0No' in helmet_info:
        helmet_info_encoding[0] = '1'

    if 'P1' in helmet_info:
        helmet_info_encoding[1] = '1'
        helmet_info_encoding[6] = '1'
    if 'P1No' in helmet_info:
        helmet_info_encoding[1] = '1'

    if 'P2' in helmet_info:
        helmet_info_encoding[2] = '1'
        helmet_info_encoding[7] = '1'
    if 'P2No' in helmet_info:
        helmet_info_encoding[2] = '1'

    if 'P3' in helmet_info:
        helmet_info_encoding[3] = '1'
        helmet_info_encoding[8] = '1'
    if 'P3No' in helmet_info:
        helmet_info_encoding[3] = '1'
    return "".join(helmet_info_encoding)



for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.csv'):
            label_path = root_dir + '/' + file
            df = pd.read_csv(label_path)
            df.sort_values(by='frame_id', inplace=True)
            now_frame_id = -1
            now_txt_dir = ''
            img_path = ''
            now_helmet_label_dir = ''
            for _ , row in df.iterrows():
                if(now_frame_id is not int(row[1])):
                    now_frame_id = row[1]
                    img_idx += 1
                    img_path = img_dir + '/' + file.split('.')[0] + '/' + str(now_frame_id).zfill(2) + '.jpg'
                    shutil.copy(img_path, new_img_dir + '/' + str(img_idx).zfill(2) + '.jpg')
                    now_txt_dir = new_label_dir + '/' + str(img_idx).zfill(2) + '.txt'
                    now_helmet_label_dir = new_helmet_label_dir + '/' + str(img_idx).zfill(2) + '.txt'
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
                f = open(now_helmet_label_dir,'a')
                helmet_label = row[6]
                helmet_info = helmet_label.split('Helmet')
                helmet_info_encoding = encodeHelmet(helmet_info)
                f.write(helmet_info_encoding + ' ' + str(x) + ' '+ str(y) + ' '+ str(w) + ' '+ str(h) + '\n')
                f.close()

