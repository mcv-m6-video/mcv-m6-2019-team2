from typing import Optional
from .groundtruth import GroundTruth
from .frameExtraction import getgroundTruth
import numpy as np


class GroundTruths:

    listGd: list[GroundTruth]

    def __init__(self,listGd=[]):
        self.listGd=listGd

    def getgroundTruth(self,directory_txt):
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
            bb = GroundTruth(
                frameid,
                topleft,
                width,
                height,
                confidence)
            self.listGd.append(bb)
        txt_gt.close()

    def modify_random_gt(self, prob):
        rd = np.random.choice([0, 1], size=(len(self.listGd),), p=[1 - prob, prob])
        for i in enumerate(rd):
            if (rd[i]):
                self.listGd[i].modify_gt()

    def eliminate_random_gt(self, prob):
        rd=np.random.choice([0,1],size=(len(self.listGd),), p=[1-prob,prob])
        for i in enumerate(rd):
            if(rd[i]):
                self.listGd[i].remove()











