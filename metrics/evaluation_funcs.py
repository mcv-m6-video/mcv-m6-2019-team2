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
def iou_frame(gt_frame:Frame,detections_frames:Frame,thres):
    TP=0
    iouframe=[]
    u=0
    i=0
    for u in gt_frame.bboxes:
        """bboxes=detections_frames.get_bboxes()
        for i in bboxes:"""
        for i in detections_frames.bboxes:
            #print(u)
            #print(i)
            if(u.iou(i)> thres):
                TP+=1
                iouframe.append(u.iou(i))
    return iouframe, TP

def iou_video(gt:Video, detections:Video, thres=0.1):
    TP = 0
    iou_frm=[]
    for i in gt.list_frames:
            frame_detec = detections.get_frame_by_id(i.frame_id)
            #print(frame_detec.bboxes)

            ioufrm,TP_fr = iou_frame(i, frame_detec, thres)
            TP+=TP_fr

    return TP


def iou_TFTN_video(gt:Video, detections:Video, thres=0.1):
    TP=0
    FP=0
    FN=0

    TP =iou_video(gt, detections, thres)

    FP = len(detections.get_detections_all())-TP

    FN = len(gt.get_detections_all())-TP

    return TP,FP,FN

def iou_overtime(gt:Video, detections:Video, thres=0.1):
    iou_by_frame=[]
    for i in gt.list_frames:
        iouframe,TP=iou_frame(i, detections.get_frame_by_id(i.frame_id), thres)
        if len(iouframe) > 1:
            iou_mean = mean(iouframe)
        else:
            iou_mean = iouframe
        iou_by_frame.append(iou_mean)
    return iou_by_frame
