import numpy as np
import os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
def read_helmet_label_file(file_path):
    f = open(file_path, 'r')
    bbox_list = []
    label_list = []
    lines = f.readlines()
    for line in lines:
        temp_list = line.split()
        label = temp_list[0]
        bbox_info = temp_list[1:]
        bbox = [int(x) for x in bbox_info]
        for id in range(4):
            if label[id] == '0':
                label = label[:id+5] + '0' + label[id+6:]
        bbox_list.append(bbox)
        label_list.append(label)
    f.close()
    return label_list, bbox_list

def match_bbox(pred_bbox, gt_bbox, gt_label):
    maxMinY = max(pred_bbox[1], gt_bbox[1])
    maxMinX = max(pred_bbox[0], gt_bbox[0])
    minMaxY = min(pred_bbox[1] + pred_bbox[3], gt_bbox[1] + gt_bbox[3])
    minMaxX = min(pred_bbox[0] + pred_bbox[2], gt_bbox[0] + gt_bbox[2])
    IOU = 0
    Label = 'Not Matched'
    if not (maxMinY > minMaxY or maxMinX > minMaxX):
        intersection = (minMaxY-maxMinY)*(minMaxX-maxMinX)
        union = pred_bbox[3]*pred_bbox[2] + gt_bbox[3]*gt_bbox[2] - intersection
        IOU = intersection / union
    if IOU < 0.1:
        IOU = 0
    else:
        Label = gt_label
    return IOU, Label

def generate_confusion_matrix(gt_root, pred_root):
    gt_list = []
    pred_list = []
    for subdir, dirs, files in os.walk(pred_root):
        for file in files:
            if file.endswith('.txt'):
                pred_label_list, pred_bbox_list = read_helmet_label_file(os.path.join(pred_root, file))
                gt_label_list, gt_bbox_list = read_helmet_label_file(os.path.join(gt_root, file))
                for pred_id, pred_bbox in enumerate(pred_bbox_list):
                    maxIOU = 0
                    matchedLabel = 'Not Matched'
                    for gt_id, gt_bbox in enumerate(gt_bbox_list):
                        tempIOU, tempLabel = match_bbox(pred_bbox, gt_bbox, gt_label_list[gt_id])
                        maxIOU = max(tempIOU, maxIOU)
                        if maxIOU == tempIOU:
                            matchedLabel = tempLabel
                    gt_list.append(matchedLabel)
                    pred_list.append(pred_label_list[pred_id])
    return gt_list, pred_list

if __name__ == '__main__':
    gt_list, pred_list = generate_confusion_matrix(r'E:\DL_data\HELMET\helmet_labels', r'E:\DL_data\HELMET\inference_output')
    label_set = set.union(set(gt_list), set(pred_list))
    label_list = list(label_set)
    pred_list_label = list(set(pred_list))
    pred_list_label.append('Not Matched')
    show = confusion_matrix(gt_list, pred_list, labels=label_list)
    true_pos = np.diag(show)
    false_pos = np.sum(show, axis=0) - true_pos
    false_neg = np.sum(show, axis=1) - true_pos

    precision = (true_pos / (true_pos + false_pos))
    recall = true_pos / (true_pos + false_neg)
    precisions = []
    recalls = []
    for num in precision:
        if math.isnan(num):
            precisions.append(0)
        else:
            precisions.append(num)
    for num in recall:
        if math.isnan(num):
            recalls.append(0)
        else:
            recalls.append(num)


    for id in range(len(label_list)):
        if label_list[id] != 'Not Matched':
            label_list[id] = str(id)

    df_cm = pd.DataFrame(show, index=label_list,
                         columns=label_list)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./confusion_matrix.png')
    # plt.show()
    print('Precision: ',precisions, 'Recall: ', recalls)





