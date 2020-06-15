import cv2
import os
import numpy as np


def imgToHu(img):
    _, im = cv2.threshold(img, 128,  255, cv2.THRESH_BINARY)
    moment = cv2.moments(im)
    huMoments = cv2.HuMoments(moment)

    return huMoments.T



if __name__ == "__main__":
    srcPath = "../data/crHand/"
    sdtPath = "../data/txtHu/"
    i = 0
    while(i <= 9):
        for fileName in os.listdir(srcPath+str(i)):
            k = 0
            hu = np.empty((len(os.listdir(srcPath+str(i))),7))
            img = cv2.imread(srcPath+str(i)+"/"+fileName,0)
            huMoment = imgToHu(img)
            hu[k] = huMoment
            k= k + 1
        np.savetxt(sdtPath+str(i)+".csv" ,hu ,delimiter=",", newline='\n')
        hu = []
        i = i + 1
    # np.savetxt('../data/txtList/tmp.txt', hu, delimiter=",",newline="\n")