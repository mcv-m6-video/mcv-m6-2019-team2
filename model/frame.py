import os
import cv2
from .bbox import BBox

class Frame:
    frame_id:int
    bboxes: []
    def __init__(self,bboxes=[]):
        self.bboxes=[BBox()]

def frame_extraction_ffmpeg(source, frame_folder):

    frame_folder='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/frames'
    if not os.path.exists(frame_folder):
        os.mkdir(frame_folder)

        video_source = source + 'vdo.avi'
        frame_dest = frame_folder + '/image%d.jpg'
        command = 'ffmpeg -i ' + video_source + ' -q:v 1 ' + frame_dest + ' -hide_banner'
        os.system(command)

def frame_extraction_cv2(source, folder_frame):
    video_source = source + 'vdo.avi'
    video = cv2.VideoCapture(video_source)
    success, image = video.read()
    count = 0
    os.mkdir(folder_frame)
    while success:
        cv2.imwrite("/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/video_frame/frame%d.jpg" % count,image)  # save frame as JPEG file
        success, image = video.read()
        print('Read a new frame: ', success)
        count += 1

def iou_frame():
def show_frame():
