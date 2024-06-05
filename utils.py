import argparse
import torch.nn as nn
from medpy import metric
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""
from scipy.spatial.distance import directed_hausdorff
import cv2
import numpy as np

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize(input_image):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    # 显示每个通道的图像
    for i in range(4):
        axs[i].imshow(input_image[:, :, i], cmap='gray')  # 使用灰度色彩图显示每个通道的值
        axs[i].set_title(f'Channel {i + 1}')

    plt.show()


def dice_coefficient_in_train_camus(true_vessel_arr, pred_vessel_arr):
    diceavg = np.zeros([3, 1])
    dc = np.zeros([3, true_vessel_arr.shape[0]])
    intersection = np.zeros([true_vessel_arr.shape[0], 1])
    size1 = np.zeros([true_vessel_arr.shape[0], 1])
    size2 = np.zeros([true_vessel_arr.shape[0], 1])
    for ii in range(1, 4):
        # print(ii,true_vessel_arr.shape,pred_vessel_arr.shape)

        true_vessel_array = true_vessel_arr[:, :, :, ii].astype(np.bool)
        pred_vessel_array = pred_vessel_arr[:, :, :, ii].astype(np.bool)

        # print(ii,true_vessel_array.shape,pred_vessel_array.shape)
        for jj in range(0, true_vessel_arr.shape[0]):
            intersection[jj] = np.count_nonzero(true_vessel_array[jj, :, :] & pred_vessel_array[jj, :, :])

            size1[jj, 0] = np.count_nonzero(true_vessel_array[jj, :, :])
            size2[jj, 0] = np.count_nonzero(pred_vessel_array[jj, :, :])

            # print(ii,intersection.shape,size1.shape)
            # dc1 = 2. * intersection / float(size1 + size2)
            if size1[jj, 0] + size2[jj, 0] == 0:
                #visualize(true_vessel_arr[jj])
                #visualize(pred_vessel_arr[jj])
                dc[ii - 1, jj] = 0.0
            else:
                dc[ii - 1, jj] = 2. * intersection[jj, 0] / float(size1[jj, 0] + size2[jj, 0])

        diceavg[ii - 1, 0] = np.mean(dc[ii - 1, :])
    return diceavg

def dice_coefficient_in_train(true_vessel_arr, pred_vessel_arr):
    diceavg = np.zeros([4, 1])
    dc = np.zeros([2, true_vessel_arr.shape[0]])
    intersection = np.zeros([true_vessel_arr.shape[0], 1])
    size1 = np.zeros([true_vessel_arr.shape[0], 1])
    size2 = np.zeros([true_vessel_arr.shape[0], 1])
    for ii in range(1, 5):
        # print(ii,true_vessel_arr.shape,pred_vessel_arr.shape)

        true_vessel_array = true_vessel_arr[:, :, :, ii].astype(np.bool)
        pred_vessel_array = pred_vessel_arr[:, :, :, ii].astype(np.bool)

        # print(ii,true_vessel_array.shape,pred_vessel_array.shape)
        for jj in range(0, true_vessel_arr.shape[0]):
            intersection[jj] = np.count_nonzero(true_vessel_array[jj, :, :] & pred_vessel_array[jj, :, :])

            size1[jj, 0] = np.count_nonzero(true_vessel_array[jj, :, :])
            size2[jj, 0] = np.count_nonzero(pred_vessel_array[jj, :, :])

            # print(ii,intersection.shape,size1.shape)
            # dc1 = 2. * intersection / float(size1 + size2)
            try:
                dc[ii - 1, jj] = 2. * intersection[jj, 0] / float(size1[jj, 0] + size2[jj, 0])
            except ZeroDivisionError:
                dc[ii - 1, jj] = 0.0

        diceavg[ii - 1, 0] = np.mean(dc[ii - 1, :])
    return diceavg

def  calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        hd=0
        for i in range(1,4):
            hd95 = metric.binary.hd95(pred[:,:,:,i], gt[:,:,:,i])
            hd+=hd95
        return hd/3
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def  calculate_metric_percase_ACDC(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        hd=0
        for i in range(1,4):
            hd95 = metric.binary.hd95(pred[:,:,:,i], gt[:,:,:,i])
            hd+=hd95
        return hd/3
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
