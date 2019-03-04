from typing import Optional
from .bbox import BBox
from .frame import Frame
import numpy as np
import random


class Video:
    list_frames: list

    def __init__(self, listGd=[]):
        self.listGd = []

    def getgroundTruth(self, directory_txt):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively

        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            self.listGd.append(Frame(frameid, BBox(topleft,width,height,confidence)))
        txt_gt.close()

    def modify_random_bbox(self, prob):
        rd = np.random.choice([0, 1], size=(len(self.listGd),), p=[1 - prob, prob])
        list_mod=[]
        for i, j in enumerate(rd):
            if (rd[i]):
                noise = random.uniform(0, 0.5)
                list_mod[i].modify_gt(noise)
        return list_mod

    def eliminate_random_gt(self, prob):
        rd = np.random.choice([0, 1], size=(len(self.listGd),), p=[1 - prob, prob])
        ind = []
        list_elim=[]
        for i, j in enumerate(rd):
            if (rd[i]): ind.append(i)
        for i in reversed(ind): list_elim.pop(i)
        return list_elim

    def get_num_frames(self):
        frame = 0
        for i in self.listGd:
            if (i.frame_id != frame):
                frame = i.frame_id
        return frame

    def get_detections_by_frame(self, numb_frame):
        listatrr = BoundingBoxes_Video()
        index = []
        j = 0
        for i in self.listGd:
            j += 1
            if i.frame_id == numb_frame:
                listatrr.listGd.append(i)
                index.append(j)

        return index, listatrr
