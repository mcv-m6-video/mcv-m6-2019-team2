import cv2
import numpy as np

import math

from matplotlib import pyplot as plt


# based on flow development kit of kitti web

def flow_read(gt_dir, test_dir):

    gt_kitti = cv2.imread(gt_dir, -1)

    test_kitti = cv2.imread(test_dir, -1)

    gt_kitti.astype(float)

    test_kitti.astype(float)

    u_test = (test_kitti[:, :, 1].ravel() - math.pow(2, 15)) / 64

    v_test = (test_kitti[:, :, 2].ravel() - math.pow(2, 15)) / 64

    u_gt = (gt_kitti[:, :, 1].ravel() - math.pow(2, 15)) / 64

    v_gt = (gt_kitti[:, :, 1].ravel() - math.pow(2, 15)) / 64

    valid_gt = gt_kitti[:, :, 0].ravel()

    return u_test, v_test, u_gt, v_gt, valid_gt


gt_dir = "/Users/quim/Desktop/untitled/datasets/kitti/groundtruth/000045_10.png"

test_dir = '/Users/quim/Desktop/untitled/datasets/kitti/results/LKflow_000045_10.png'

u_test, v_test, u_gt, v_gt, valid_gt = flow_read(gt_dir, test_dir)


