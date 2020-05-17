import cv2
import random
import numpy as np

class Detector_Augment():
    def __init__(self,size=512):
        self.W,self.H = size,size

    def flip_point(self, point, operation=1):
        x, y,w,h = point

        if operation == 0:
            return np.array([x, 1 - y, w, h])
            # return np.array([x, self.H - y,w,h])
        if operation == 1:
            return np.array([1 - x, y, w, h])
            # return np.array([self.W - x, y,w,h])

    def RandomFlip(self,img,gt):
        operations = [0,1]

        for operation in operations:
            if random.random() < 0.5:
                gt = self.flip_point(gt, operation)
                img = cv2.flip(img, operation)

        return img,gt

    def show(self,img,gt):
        cv2.rectangle(img, (int(gt[0]-gt[2]/2),int(gt[1]-gt[3]/2)),(int(gt[0]+gt[2]/2),int(gt[1]+gt[3]/2)), (0, 255, 0), 2)
        cv2.imshow("img", img)
        cv2.waitKey()



if __name__=="__main__":
    img_path = "D:/Project_Glaucoma/DENet/test_image/test.jpg"

    gt = [115,265,70,90]
    img = cv2.imread(img_path)
    img = cv2.resize(img,(512,512))

    augment = Detector_Augment()
    img,gt = augment.RandomFlip(img,gt)
    augment.show(img,gt)