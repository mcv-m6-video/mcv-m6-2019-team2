import os
import cv2
import numpy as np
import math as mt
from .bbox import BBox

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
        self.readVideoBW(self.dir_path)

    def readVideoBW(self,dir_path='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010'):

        #Frame path
        frame_path=dir_path+'/framess'
        #gt path
        gt_path=dir_path+'/gt'

        frame_list = sorted(os.listdir(frame_path))

        num_frames=len(frame_list)


        j=0
        for i,j in enumerate(frame_list):
            print(i)
            if i<= mt.trunc(0.25*num_frames):
                image_path=frame_path+'/image'+str(i)+'.jpg'
                im=cv2.imread(image_path,0)
                if im is not None:
                    im_v=np.reshape(im,im.shape[0]*im.shape[1])
                    self.train_frames.append([i,im])
            else:
                image_path = frame_path + '/image' + str(i) + '.jpg'
                im = cv2.imread(image_path, 0)
                if im is not None:
                    im_v = np.reshape(im, im.shape[0] * im.shape[1])
                    self.test_frames.append([i,im])

    #def creategt(self):

    def modeltrainGaussian(self):
        mean_frames=[]
        std_frames=[]
        for i in self.train_frames:
            mean_frames.append(np.mean(i))
            std_frames.append(np.std(i))
        self.mean_train = np.mean(mean_frames)
        self.std_train = np.mean(std_frames)

    def classifyTest(self,alpha,rho,isAdaptive):
        alpha=1#????
        rho = 0.5 # Evaluate different vals
        for i in self.test_frames:
            for j in i:
                if (j+self.mean_train) >= alpha*(self.std_train+2):
                    foreground_pixel=1 #to continue

                if isAdaptive and foreground_pixel == 0: # Only background pixels
                    self.mean_train = rho * j + (1-rho)*self.mean_train
                    self.std_train = rho * (j - self.mean_train)**2 + (1 - rho) * self.std_train

    @staticmethod
    def getgt_detections(directory_txt):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        boxes_gt = []
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            boxes_gt.append(BBox(frameid,topleft ,width, height, confidence))

        return boxes_gt

    def creategt(self,frames):
        gt_dir=self.dir_path+'/gt'
        boxes_gt=self.getgt_detections(gt_dir)
        box_frame=[]
        white_image=255*np.ones_like(frames[0][1])

        for i in frames:
            for u in boxes_gt:
                if u.frame_id==i[0]:
                    box_frame[i].append(u)


        for i in frames:
            for u in box_frame[i]:
                [xmin, ymin, xmax, ymax]=u.to_result()
                roi = i[1][xmin:xmax , ymin:ymax]
                for val in np.unique(roi)[1:]:  # step 2
                    mask = np.uint8(roi == val)  # step 3
                    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
                    white_image[labels == largest_label] = val












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