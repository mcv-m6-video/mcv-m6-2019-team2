import numpy as np
from model import Video
from model import Frame
from statistics import mean

def performance_evaluation(TP, FN, FP):
    """
    performance_evaluation_window()

    Function to compute different performance indicators (Precision, accuracy, 
    sensitivity/recall) at the object level
    
    [precision, sensitivity, accuracy] = PerformanceEvaluationPixel(TP, FN, FP)
    
       Parameter name      Value
       --------------      -----
       'TP'                Number of True  Positive objects
       'FN'                Number of False Negative objects
       'FP'                Number of False Positive objects
    
    The function returns the precision, accuracy and sensitivity
    """
    
    precision   = float(TP) / float(TP+FP); # Q: What if i do not have TN?
    sensitivity = float(TP) / float(TP+FN)
    accuracy    = float(TP) / float(TP+FN+FP);

    return [precision, sensitivity, accuracy]

def iou_video(gt:Video, detections:Video, thres=0.1):
    TP = 0
    iou_frame=[]
    for i in gt.list_frames:
            frame_detec = detections.get_frame_by_id(i.frame_id)

            TP_fr,ioufrm = iou_frame(i, frame_detec, thres)
            #TP+=TP_fr

    return TP

def iou_frame(gt_frame:Frame,detections_frames:Frame,thres):
    TP=0
    iouframe=[]
    for i in gt_frame.bboxes:
        for j in detections_frames.bboxes:
            if(i.iou(j)> thres):
                TP+=1
                iouframe.append(i.iou(j))
    return iouframe, TP


def iou_TFTN_video(gt:Video, detections:Video, thres=0.1):
    TP=0
    FP=0
    FN=0

    TP, iou_frame =iou_video(gt, detections, thres)

    FP = len(detections.get_detections_all())-TP

    FN = len(gt.get_detections_all())-TP

    return TP,FP,FN

def iou_overtime(gt:Video, detections:Video, thres=0.1):
    iou_by_frame=[]
    for i in gt.list_frames:
        TP_fr,iou_frame=iou_frame(i, detections.get_frame_by_id(i.frame_id), thres)
        if len(iou_frame) > 1:
            iou_mean = mean(iou_frame)
        else:
            iou_mean = iou_frame
        iou_by_frame.append(iou_mean)
    return iou_by_frame
