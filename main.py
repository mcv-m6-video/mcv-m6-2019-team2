from opticalFlow import Optical_flow
from model.read_annotation import read_annotations
from model import *
from metrics import *

def task0():
    """After annotate between frames 391-764 we convert cvat xml to ai city challenge format """

    file = 'annotations/AI_CITY_S03_C01_391_764.xml'
    read_annotations(file)

def task11():
    """Implement IoU and mAP:
    -Add noise to change size and position of bounding boxes
    -Add probability to generate/delete bounding boxes
    +Analysis & Evaluation"""
    gt_dir='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/annotation.txt'
    det_dir='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_ssd512.txt'
    det_dir2 = '/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'

    #Apply modifications and eliminate samples of the gt given.
    gt_video = Video(Video().getgroundTruthown(gt_dir,300))
    print('hola')
    gt_video_modif1=Video(Video().getgt_detections(det_dir2, 1000))
    gt_video_modif2=Video(Video().getgt_detections(det_dir, 1000 ))
    print(len(gt_video_modif1.get_detections_all()))
    print('hey')
    #Apply modifications and eliminate samples of the gt given

    #First modification:
    # modify randomnly in range of 20% the bounding boxes
    gt_video_modif1.modify_random_bboxes(0)
    # eliminate the randomnly 40% of the bounding boxes
    gt_video_modif1.eliminate_random_bboxes(0)

    #Second modification:
    # modify randomnly in range of 20% the bounding boxes
    gt_video_modif2.modify_random_bboxes(0)
    # eliminate the randomnly 40% of the bounding boxes
    gt_video_modif2.eliminate_random_bboxes(0)

    #Evaluation

    #IOU global
    TP1, FP1, FN1 = iou_TFTN_video(gt_video, gt_video_modif1)
    [precision1, sensitivity1, accuracy1] = performance_evaluation(TP1, FN1, FP1)

    TP2, FP2, FN2 = iou_TFTN_video(gt_video, gt_video_modif2)
    [precision2, sensitivity2, accuracy2] = performance_evaluation(TP2, FN2, FP2)
    print('hola')
def task12():
    #mAP(mean average)
    [mAP1,precisions1,recalls1, average_precision1]=app_accumulator(gt_video, gt_video_modif1,0.5)
    [mAP2,precisions2,recalls2, average_precision2]=app_accumulator(gt_video, gt_video_modif2,0.5)
    #ax=plt.plot()
    #mAP1.plot_pr(ax, 1, precisions1, recalls1, average_precision1)

    mAP1.plot(precisions1, recalls1, average_precision1)
    plt.show()
    mAP2.plot(precisions2, recalls2, average_precision2)
    plt.show()
    #display results







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
    task11()
    #task2()
    #task3(157) # 45 or 157
    #task4()
