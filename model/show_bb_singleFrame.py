from metrics.evaluation_funcs import iou_overtime
from model import *
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from PIL import Image
import numpy as np

def show_bboxes(path, bboxes: Frame, bboxes_noisy: Frame):
    """
    shows the ground truth and the noisy bounding boxes
    :param path:
    :param bboxes:
    :param bboxes_noisy:
    :return:
    """

    im = np.array(Image.open(path), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for bbox in bboxes.bboxes:

        rect = patches.Rectangle((bbox.top_left),
                                 bbox.width , bbox.height ,
                             linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    for bbox_noisy in bboxes_noisy.bboxes:
        bb = bbox_noisy.to_result()
        rect = patches.Rectangle((bbox_noisy.top_left),
                                 bbox_noisy.width , bbox_noisy.height,
                             linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)
    # Add the patch to the Axes


    plt.show()




path= '/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/frames/image391.jpg'

gt_dir = 'annotation.txt'
gt_video = Video(Video().getgroundTruthown(gt_dir, 350))
gt_video_modif1 = Video(Video().getgroundTruthown(gt_dir,350))

gt_video_modif1.modify_random_bboxes(0.2)


detections_bboxes = gt_video_modif1.get_frame_by_id(391)
gt_bboxes = gt_video.get_frame_by_id(391)

show_bboxes(path,gt_bboxes,detections_bboxes)