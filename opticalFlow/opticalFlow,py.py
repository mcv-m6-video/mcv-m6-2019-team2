import cv2
import numpy as np
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import math

import configuration as conf

def visualizeOF():
    gtImg = sorted(glob.glob(conf.gtFolder + "*png"))

    for nImg in range(len(gImgtt)):
		# load img unchanged
        img = cv2.imread(gtImg[nImg], -1) 
		cv2.imshow('img', img)
		
		# Down-sample img by applying function to local blocks (img, block_size, func=<function sum>, cval=0)
        imgScaled = block_reduce(img, block_size=(2,2,1), func=np.mean)
        row, col, channels = imgScaled.shape;
		
		# init vars
        uFlow = []
        vFlow = []
        validGT= []

		# for all pix in img
        for i in range(OF_flow.shape[0]):
			for j in range(OF_flow.shape[1]):
    			isOF = imgScaled[i,j,0]
				if isOF == 1:
					uComputeDif = (float)(imgScaled[i,j,0]) - math.pow(2, 15)
					uFlow.append(ucomputeDif / 64.0)
					vComputeDif = (float)(imgScaled[i,j,1]) - math.pow(2, 15)
					vFlow.append(vcomputeDif / 64.0)
				else:
					uFlow.append(0)
					vFlow.append(0)
				validGT.append(isOF)

		#reshape  to the Scaled size
        uFlow = np.reshape(uFlow, (row, col))
        vFlow = np.reshape(vFlow, (row, col))
		
        x, y = np.meshgrid(np.arange(0, col, 1), np.arange(0, row, 1))

		#  SCALE = units per arrow lenght , ALPTHA = transparences	
		#  X and Y set the location of the arrows, U and V are the arrow data
        plt.quiver(x, y, uFlow, vFlow, scale=1,  alpha = 0.5, linewidth = 0.01)
        plt.show()
        

