import os
import cv2
import numpy as np
import math as mt

class OneGaussianVideo:

    train_frames=list
    test_frames=list
    dir_path=str
    mean_train:float
    std_train:float

    def __init__(self, train_frames=[],test_frames=[],dir_path='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010',mean_train=0,std_train=0):
        self.dir_path=dir_path
        self.train_frames = []
        self.test_frames=[]
        self.mean_train=0
        self.std_train=0
        self.readVideoBW(dir_path)

    def readVideoBW(self,dir_path='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010'):

        #Frame path
        frame_path=dir_path+'/framep'
        #gt path
        gt_path=dir_path+'/gt'

        frame_list = sorted(os.listdir(frame_path))

        num_frames=len(frame_list)
        j=0
        for i in frame_list:
            j = +1
            if j< mt.trunc(0.75*num_frames):
                if i[0]==".":
                    pass
                else:
                    im=cv2.imread(frame_path+"/"+i,0)
                    im_v=np.reshape(im,im.shape[0]*im.shape[1])
                    self.train_frames.append(im_v)
            else:
                im = cv2.imread(frame_path + "/" + i, 0)
                im_v = np.reshape(im, im.shape[0] * im.shape[1])
                self.test_frames.append(im_v)

    #def creategt(self):

    def modeltrainGaussian(self):
        mean_frames=[]
        std_frames=[]
        for i in self.train_frames:
            mean_frames.append(np.mean(i))
            std_frames.append(np.std(i))
        self.mean_train = np.mean(mean_frames)
        self.std_train = np.mean(std_frames)

    def classifyTest(self):
        alpha=1#????
        for i in self.test_frames:
            for j in i:
                if (j+self.mean_train) >= alpha*(self.std_train+2):
                    foreground_pixel=1 #to continue














"""
    @staticmethod
    def getgt_detections(directory_txt, num_frames):
       """ """Read txt files containing bounding boxes (ground truth and detections).""""""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        vid_fr = []
        frameid_saved = 1
        Boxes_list = []
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            Boxes_list.append(('frameid':frameid, 'topleft':topleft, 'width': width, 'height':height, 'confidence':confidence))

        for i in range(0, num_frames):
            items = [item for item in Boxes_list if item.frame_id == i]
            if items:
                vid_fr.append(Frame(i, items))

            txt_gt.close()
        return vid_fr"""