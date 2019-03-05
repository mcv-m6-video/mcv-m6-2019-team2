from Optical_flow import *
from read_annotation import read_annotations


def task0():
    """After annotate between frames 391-764 we convert cvat xml to ai city challenge format """

    file = 'annotations/AI_CITY_S03_C01_391_764.xml'
    read_annotations(file)

def task1():
    """Implement IoU and mAP:
    -Add noise to change size and position of bounding boxes
    -Add probability to generate/delete bounding boxes
    +Analysis & Evaluation"""

def task2():
    """Temporal analysis: IOU overtime """


def task3(seq):
    """Optical flow: Numerical result for MSEN and PEPN, histogram error and error visualization"""

    # change the number of sequence to visualize the results for both cases

    if seq == 45:
        gt_dir = "datasets/kitti/groundtruth/000045_10.png"
        test_dir = 'datasets/kitti/results/LKflow_000045_10.png'
    if seq == 157:
        gt_dir = "datasets/kitti/groundtruth/000157_10.png"
        test_dir = 'datasets/kitti/results/LKflow_000157_10.png'

    F_gt, F_test = flow_read(gt_dir, test_dir)

    MSEN = msen(F_gt, F_test)
    PEPN = pepn(F_gt, F_test, 3)

    print(MSEN)
    print(PEPN)

def task4():
    """optical flow visualization"""


if __name__ == '__main__':
    task0()
    task1()
    task2()
    task3(157) # 45 or 157
    task4()
