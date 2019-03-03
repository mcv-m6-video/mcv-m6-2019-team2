import numpy as np
from model import BoundingBoxes_Video
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

def iou_gt(gt:BoundingBoxes_Video, detections:BoundingBoxes_Video, thres=0.1):
    TP = 0
    iou_frames=[]
    for i in gt.listGd:
        framegt=[]
        for x in detections.listGd:
           if x.frame_id ==i.frame_id:
               framegt.append(x)
        if framegt:
            iou_frame=[]
            for j in framegt:
                iou_frame.append(i.iou(j))


            iou_frames.append(max(iou_frame))
                #print(i.iou(j))
            if (max(iou_frame) > thres):
                TP = TP + 1
    return TP, iou_frames

def iou_TFTN(gt:BoundingBoxes_Video, detections:BoundingBoxes_Video, thres=0.1):
    TP=0
    FP=0
    FN=0

    TP, iou_frame =iou_gt(gt, detections, thres)

    FP = len(detections.listGd)-TP

    FN = len(gt.listGd)-TP

    return TP,FP,FN

def iou_overtime(gt:BoundingBoxes_Video,detections:BoundingBoxes_Video,thres=0.1):
    num_frames=gt.get_num_frames()
    iou_by_frame=[]
    for i in range(0,num_frames):
        [index, listatrr1]=gt.get_detections_by_frame(i)
        [index, listatrr2] = detections.get_detections_by_frame(i)
        TP, iou_frame = iou_gt(listatrr1, listatrr2, thres)
        if len(iou_frame)>1:
            iou_mean=mean(iou_frame)
        else:
            iou_mean=iou_frame
        iou_by_frame.append(iou_mean)
    return iou_by_frame
