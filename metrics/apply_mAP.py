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


#source = '/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/'
source = '/home/arnau/Documents/Master/M6/mcv-m6-2019-team2/datasets/train/S03/c010/'
video_source = source + 'vdo.avi'

#folder_frame ='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/video_frame'
folder_frame ='/home/arnau/Documents/Master/M6/mcv-m6-2019-team2/datasets/train/S03/c010/video_frame'

#dir_gt='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'
dir_gt='/home/arnau/Documents/Master/M6/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'
#frame_extraction_ffmpeg(source, folder_frame)

#getgroundTruth(dir_gt)

vid=Video().getgroundTruth(dir_gt,10)
video=Video(Video().getgroundTruth(dir_gt,10))
video.modify_random_bboxes(0.2)

video.eliminate_random_bboxes(0.4)
vido=Video(Video().getgroundTruth(dir_gt,10))
#print(len(bb2.listGd))


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

for i in range(video.get_num_frames()):
    for j in range(len(video.list_frames[i].bboxes)):
        pred_bb.append(video.list_frames[i].bboxes[j].to_result())
        pred_conf.append(video.list_frames[i].bboxes[j].get_condidence())
        pred_classes.append(1) # all same class
    for j in range(len(vido.list_frames[i].bboxes)):
        gt_bb.append(vido.list_frames[i].bboxes[j].to_result())
        gt_classes.append(1)

#frames = [pred_bb, pred_classes, pred_conf, gt_bb, gt_classes]

n_class = 1
mAP = DetectionMAP(n_class)
for i in range(video.get_num_frames()):
    #print("Evaluate frame {}".format(i))
    show_frame(np.array(pred_bb), np.array(pred_classes),
               np.array(pred_conf), np.array(gt_bb), np.array(gt_classes))
    mAP.evaluate(np.array(pred_bb), np.array(pred_classes),
                 np.array(pred_conf), np.array(gt_bb), np.array(gt_classes))

mAP.plot()
plt.show()

#n_class = 1
#mAP = DetectionMAP(n_class)
#for i, frame in enumerate(frames):
#    print("Evaluate frame {}".format(i))
#    show_frame(*frame)
#    mAP.evaluate(*frame)
#
#mAP.plot()
#plt.show()