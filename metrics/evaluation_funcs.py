import numpy as np
from model import GroundTruths


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



def iou_gt(gt:GroundTruths, detections:GroundTruths, thres=0.1):
    TP=0
    FP=0
    FN=0
    for i in gt.listGd:
        framegt=[]
        for x in detections.listGd:
           if x.frame_id ==i.frame_id:
               framegt.append(x)

        if framegt:
            for j in framegt:
                #print(i.iou(j))
                if(i.iou(j)!=0):
                    if (i.iou(j)> thres):
                        TP=TP+1
                    else:
                        FP=FP+1
        else:
            FN=FN+1

    return TP,FP,FN

