from opticalFlow import Optical_flow
from model.read_annotation import read_annotations
from model import *
from metrics import *
import matplotlib.pyplot as plt

from opticalFlow.Optical_flow import pepn, msen, flow_read


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

    #Apply modifications and eliminate samples of the gt given.
    gt_video = Video(Video().getgroundTruthown(gt_dir,391))
    gt_video_modif1=Video(Video().getgroundTruthown(gt_dir, 391))
    gt_video_modif2=Video(Video().getgroundTruthown(gt_dir, 391 ))
    #Apply modifications and eliminate samples of the gt given

    #First modification:
    # modify randomnly the bounding boxes by 1%
    gt_video_modif1.modify_random_bboxes(0.4)
    # eliminate the randomnly 20% of the bounding boxes
    gt_video_modif1.eliminate_random_bboxes(0.1)

    #Second modification:
    # modify randomnly the bounding boxes by 1%
    gt_video_modif2.modify_random_bboxes(0.2)
    # eliminate the randomnly 40% of the bounding boxes
    gt_video_modif2.eliminate_random_bboxes(0.7)

    #Evaluation

    #IOU global
    TP1, FP1, FN1 = iou_TFTN_video(gt_video, gt_video_modif1,thres=0.5)
    [precision1, sensitivity1, accuracy1] = performance_evaluation(TP1, FN1, FP1)

    TP2, FP2, FN2 = iou_TFTN_video(gt_video, gt_video_modif2,thres=0.5)
    [precision2, sensitivity2, accuracy2] = performance_evaluation(TP2, FN2, FP2)


    print("\nFist modification:"
          "\nPrecision:",precision1,
          "\nAccuracy:",accuracy1,
          "\nSensitivity:",sensitivity1)

    print("\nSecond modification:"
          "\nPrecision:", precision2,
          "\nAccuracy:", accuracy2,
          "\nSensitivity:", sensitivity2)

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
    gt_dir = '/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/annotation.txt'
    yolo = '/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'
    ssd='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_ssd512.txt'
    rcnn='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_mask_rcnn.txt'
    gt_video = Video(Video().getgroundTruthown(gt_dir,391))
    gt_video_modif1=Video(Video().getgroundTruthown(gt_dir, 391))
    gt_video_modif2=Video(Video().getgroundTruthown(gt_dir, 391))

    # First modification:
    # modify randomnly the bounding boxes by 1%
    gt_video_modif1.modify_random_bboxes(0.1)

    # iou-gt
    iou_by_frame=iou_overtime(gt_video,gt_video_modif1, thres = 0.5)


    num_frames=len(iou_by_frame)
    plt.plot(iou_by_frame)
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IOU-overtime:GT modified')
    axes=plt.gca()
    axes.set_ylim(0,1)
    axes.set_xlim(0,num_frames)
    plt.show()

    # iou_overtime -yolo
    yolo_video = Video(Video().getgroundTruthown(yolo, 391))
    iou_by_frame_yolo = iou_overtime(gt_video,yolo_video,thres=0.8)
    num_framesyolo=len(iou_by_frame_yolo)
    plt.plot(iou_by_frame_yolo)
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IOU-overtime:YOLO')
    axes=plt.gca()
    axes.set_ylim(0,1)
    axes.set_xlim(0,num_framesyolo)
    plt.show()

    # iou_overtime -ssd
    ssd_video = Video(Video().getgroundTruthown(ssd, 391))
    #eliminate some frames detections
    #yolo_video.modify_random_bboxes(0.05)
    iou_by_frame_ssd = iou_overtime(yolo_video,ssd_video,thres=0.5)
    num_framesssd=len(iou_by_frame_ssd)
    plt.plot(iou_by_frame_yolo)
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IOU-overtime:SSD')
    axes=plt.gca()
    axes.set_ylim(0,1)
    axes.set_xlim(0,num_framesssd)
    plt.show()

    # iou_overtime -rccn
    rcnn_video = Video(Video().getgroundTruthown(rcnn, 391))
    iou_by_frame_rcnn = iou_overtime(rcnn_video,gt_video, thres=0.5)
    num_framesrcnn=len(iou_by_frame_rcnn)
    plt.plot(iou_by_frame_rcnn)
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IOU-overtime:RCNN')
    axes=plt.gca()
    axes.set_ylim(0,1)
    axes.set_xlim(0,num_framesrcnn)
    plt.show()

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
    #task0()
    #task11()
    #task12()
    task2()
    #task3(157) # 45 or 157
    #task4()
