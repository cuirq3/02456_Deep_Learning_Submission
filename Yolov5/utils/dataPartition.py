import os
import random

root_dir = r'E:\DL_data\osfstorage-archive\HELMET_DATASET\images'
label_root = r'E:\DL_data\osfstorage-archive\HELMET_DATASET\labels'
val_proportion = 0.2
all_path = []

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.jpg'):
            all_path.append(file)

random.shuffle(all_path)
val_file = all_path[:int(len(all_path)*val_proportion):]
train_file = all_path[int(len(all_path)*val_proportion):]

os.makedirs(os.path.join(root_dir, 'train'))
os.makedirs(os.path.join(root_dir, 'val'))

os.makedirs(os.path.join(label_root, 'train'))
os.makedirs(os.path.join(label_root, 'val'))

for file in val_file:
    os.rename(os.path.join(root_dir, file), os.path.join(root_dir, 'val', file))
    os.rename(os.path.join(label_root, file.split('.')[0] + '.txt'), os.path.join(label_root, 'val', file.split('.')[0] + '.txt'))

for file in train_file:
    os.rename(os.path.join(root_dir, file), os.path.join(root_dir, 'train', file))
    os.rename(os.path.join(label_root, file.split('.')[0] + '.txt'), os.path.join(label_root, 'train', file.split('.')[0] + '.txt'))