# https://github.com/MathGaron/mean_average_precision/tree/master/mean_average_precision
import sys
sys.path.append('../')
from metrics.detection_map import DetectionMAP
from utils.show_frame import show_frame
import numpy as np
import matplotlib.pyplot as plt
from metrics.evaluation_funcs import iou_overtime, iou_TFTN_video

from model.video import *
from model import bbox


def app_accumulator(gt_video:Video, detections_video:Video,overlap_threshold, pr_samples=11):

    """
    FORMAT: frames = [pred_bb, pred_cls, pred_conf, gt_bb, gt_cls]

    :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, x2, y2] :     Shape [n_pred, 4]
    :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
    :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
    :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, x2, y2] :  Shape [n_gt, 4]
    :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
    """

    pred_bb = []
    pred_classes = []
    pred_conf = []
    gt_bb = []
    gt_classes = []


    for i in range(detections_video.get_num_frames()):
        for j in range(len(detections_video.list_frames[i].bboxes)):
            pred_bb.append(detections_video.list_frames[i].bboxes[j].to_result())
            pred_conf.append(detections_video.list_frames[i].bboxes[j].get_confidence())
            pred_classes.append(1)  # all same class
        for j in range(len(gt_video.list_frames[i].bboxes)):
            gt_bb.append(gt_video.list_frames[i].bboxes[j].to_result())
            gt_classes.append(1)  # all same class

    n_class = 2
    mAP = DetectionMAP(n_class, pr_samples, overlap_threshold )
    for i in range(detections_video.get_num_frames()):
        mAP.evaluate(np.array(pred_bb), np.array(pred_classes),
                 np.array(pred_conf), np.array(gt_bb), np.array(gt_classes))
    
    mean_average_precision = []
    precisions, recalls = mAP.compute_precision_recall_(1) # Class index
    average_precision = mAP.compute_ap(precisions, recalls)
    mean_average_precision.append(average_precision)

    print(mean_average_precision)

    return mAP,precisions,recalls, average_precision
#plt.suptitle("Mean average precision : {:0.2f}".format(sum(mean_average_precision)/len(mean_average_precision)))



#mAP.plot()
#plt.show()

#n_class = 1
#mAP = DetectionMAP(n_class)
#for i, frame in enumerate(frames):
#    print("Evaluate frame {}".format(i))
#    show_frame(*frame)
#    mAP.evaluate(*frame)
#
#mAP.plot()
#plt.show()"""