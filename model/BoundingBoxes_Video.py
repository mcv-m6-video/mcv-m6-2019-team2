from typing import Optional
from .BoundingBoxes import BoundingBoxes
import numpy as np
import random

class BoundingBoxes_Video:

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
            self.listGd.append(BoundingBoxes(frameid,topleft,width,height,confidence))
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

    def get_num_frames(self):
        frame=0
        for i in self.listGd:
            if (i.frame_id!=frame):
                frame=i.frame_id
        return frame

    def get_detections_by_frame(self, numb_frame):
        listatrr=BoundingBoxes_Video()
        index=[]
        j=0
        for i in self.listGd:
            j+=1
            if i.frame_id==numb_frame:
                listatrr.listGd.append(i)
                index.append(j)

        return index, listatrr














