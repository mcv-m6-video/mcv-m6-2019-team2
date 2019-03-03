from typing import Optional
from .groundtruth import GroundTruth
import numpy as np
import random

class GroundTruths:

    listGd: list

    def __init__(self,listGd=[]):
        self.listGd=[]

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
            self.listGd.append(GroundTruth(frameid,topleft,width,height,confidence))
        txt_gt.close()

    def modify_random_gt(self, prob):
        rd = np.random.choice([0, 1], size=(len(self.listGd),), p=[1 - prob, prob])

        for i,j in enumerate(rd):
            if (rd[i]):
                noise=random.uniform(0,0.5)
                self.listGd[i].modify_gt(noise)

    def eliminate_random_gt(self, prob):
        rd=np.random.choice([0,1],size=(len(self.listGd),), p=[1-prob,prob])
        ind=[]
        for i,j in enumerate(rd):
            if(rd[i]):ind.append(i)
        for i in reversed(ind):self.listGd.pop(i)














