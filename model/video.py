from typing import Optional
from .bbox import BBox
from .frame import Frame
import numpy as np
import random


class Video:
    list_frames: list

    def __init__(self, list_frames=[]):
        self.list_frames=list_frames

    @staticmethod
    def getgroundTruth(directory_txt):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        vid_fr=[]
        frameid_saved = -1
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            if (frameid != frameid_saved):
                vid_fr.append(Frame(frameid, BBox(topleft, width, height, confidence)))
            else:
                for i in vid_fr:
                    if (i.frame_id == frameid):
                        i.add_bbox(BBox(topleft, width, height, confidence))

            frameid_saved = frameid
        txt_gt.close()
        return vid_fr

    def modify_random_bboxes(self, prob):
        listbbox = self.list_frames
        rd = np.random.choice([0, 1], size=(len(listbbox),), p=[1 - prob, prob])
        for i, j in enumerate(rd):
            if (rd[i]):
                noise = random.uniform(0, 0.5)
                listbbox[i].modify_bbox_frame(noise)
        return listbbox

    def eliminate_random_bboxes(self, prob):
        listbbox = self.list_frames
        rd = np.random.choice([0, 1], size=(len(listbbox),), p=[1 - prob, prob])
        ind = []
        for i, j in enumerate(rd):
            if (rd[i]): ind.append(i)
        for i in reversed(ind): listbbox.pop(i)
        return listbbox

    def get_num_frames(self):
        num_frames = len(self.list_frames)
        return num_frames

    def get_frame_by_id(self, id):
        frame_r = Frame(frame_id=id, )
        index = []
        j = 0
        for i in self.list_frames:
            j += 1
            if i.frame_id == id:
                frame_r.add_bbox(i)
        return frame_r
    def get_detections_all(self):
        listbbox = Video()
        index = []
        j = 0
        for i in self.list_frames:
            for j in i.bboxes:
                listbbox.append(j)
        return listbbox
