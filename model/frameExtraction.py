import os
import cv2
from model import GroundTruth

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

def getgroundTruth(directory_txt):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    # Read GT detections from txt file
    # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
    boundingBoxes=[]
    txt_gt = open(directory_txt, "r")
    for line in txt_gt:
        splitLine = line.split(",")
        frameid = int(splitLine[0])
        topleft = [float(splitLine[2]), float(splitLine[3])]
        width = float(splitLine[4])
        height = float(splitLine[5])
        confidence = float(splitLine[6])
        bb = GroundTruth(
            frameid,
            topleft,
            width,
            height,
            confidence)
        boundingBoxes.append(bb)
    txt_gt.close()
    return boundingBoxes




source = '/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/'

video_source = source + 'vdo.avi'

folder_frame ='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/video_frame'

dir_gt='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'
#frame_extraction_ffmpeg(source, folder_frame)

getgroundTruth(dir_gt)