import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

transform = transforms.Compose([
    transforms.ToPILImage(),
    # resize
    transforms.Resize((64, 64)),
    # to-tensor
    transforms.ToTensor(),
    # normalize
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class HelmetDataset(Dataset):
    def __init__(self, root, isTrain):
        super(Dataset, self).__init__()
        self.image_root = ''
        if isTrain:
            self.image_root = os.path.join(root, 'images','train')
        else:
            self.image_root = os.path.join(root, 'images', 'val')
        self.label_root = os.path.join(root, 'helmet_labels')
        self.labels = []
        for subdir, dirs, files in os.walk(self.image_root):
            for file in files:
                if file.endswith('.jpg'):
                    f = open(os.path.join(self.label_root, file.split('.')[0] + '.txt'))
                    lines = f.readlines()
                    for line in lines:
                        item = line.split()
                        item.append(file)
                        self.labels.append(item)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = os.path.join(self.image_root, str(self.labels[index][5]))
        image = cv2.imread(path)
        box = self.labels[index][1:5]
        image = image[int(box[1]):int(box[1])+int(box[3]), int(box[0]):int(box[0])+int(box[2])]
        image = transform(image)
        helmet_info = self.labels[index][0]
        label = []
        for perChar in helmet_info:
            label.append(int(perChar))
        label = torch.tensor(np.array(label), dtype=torch.float32)
        image = torch.Tensor(image)
        return image, label, path

if __name__ == '__main__':
    dataset = HelmetDataset(r'E:\DL_data\HELMET', True)
    for idx, item in enumerate(dataset):
        image, label = item
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
    pass


