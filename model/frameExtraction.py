import os
import cv2
from model import BoundingBoxes_Video
from model import BoundingBoxes
from metrics import *
# need to be installed "brew install ffmpeg"

def frame_extraction_ffmpeg(source, folder_frame):

    """videos = os.listdir(source)
    for video in videos:
        frame_folder = folder_frame + video.split('.avi')[0]"""
    frame_folder='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/frames'
    if not os.path.exists(frame_folder):
        os.mkdir(frame_folder)

        video_source = source + 'vdo.avi'
        frame_dest = frame_folder + '/image%d.jpg'
        command = 'ffmpeg -i ' + video_source + ' -q:v 1 ' + frame_dest + ' -hide_banner'
        os.system(command)

def frame_extraction_cv2(source, folder_frame):
    video = cv2.VideoCapture(video_source)
    success, image = video.read()
    count = 0
    os.mkdir(folder_frame)
    while success:
        cv2.imwrite("/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/video_frame/frame%d.jpg" % count,image)  # save frame as JPEG file
        success, image = video.read()
        print('Read a new frame: ', success)
        count += 1






source = '/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/'

video_source = source + 'vdo.avi'

folder_frame ='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/video_frame'

dir_gt='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'
dir_gt='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'
frame_extraction_ffmpeg(source, folder_frame)
"""
#getgroundTruth(dir_gt)

bb=BoundingBoxes_Video()
bb.getgroundTruth(dir_gt)
#print(len(bb.listGd))
bb.modify_random_gt(0.2)
bb.eliminate_random_gt(0.2)
print(len(bb.listGd))
bb2=BoundingBoxes_Video()
bb2.getgroundTruth(dir_gt)
#print(len(bb2.listGd))

list=iou_overtime(bb,bb2)
print(len(list))"""
"""[TP,FP,FN]=iou_gt(bb2,bb)
[precision, sensitivity, accuracy]= performance_evaluation(TP, FN, FP)
print(TP)
print(FP)
print(FN)
print('Precision:',precision)
print('sensitivity:',sensitivity)
print('accuracy:',accuracy)"""


