import cv2
import os
import numpy as np
from utils.transforms import Detector_Augment
from torchvision import transforms
from torch.utils.data import Dataset

class Data(Dataset):

    def __init__(self,root,dataset,transform,train=True,target_size=224,Module_Type="CLS_BOX"):
        self.dataset = dataset
        self.root = root
        self.train = train
        self.transform  = transform
        self.target_size = target_size
        self.Module_Type = Module_Type
        self.detector_augment = Detector_Augment()

    def __getitem__(self, index):
        data = self.dataset[index]
        self.img = cv2.imread(os.path.join(self.root,data['img'])).astype(np.float32)/255.0
        W, H, _ = self.img.shape
        # self.img = cv2.resize(self.img, (1024, 1024))
        # if self.train:
        #     self.img = self.transform(self.img)

        # self.label = np.eye(2)[data["label"]]
        self.label = data["label"]
        if len(data["bbox"])==0:
            x,y,w,h = 0,0,0,0
        else:
            x,y,w,h = data["bbox"]
        self.gt = np.array([x/W, y/H, w/W, h/H])

        if self.Module_Type == "Global":
            pass
        elif self.Module_Type == "Local":
            w,h = w*1.5, h*1.5
            self.img = self.img[:,(x-w/2):x+w/2, (y-h/2):(y+h/2)]
        self.img = cv2.resize(self.img, (self.target_size,self.target_size))

        if self.train is True:
            self.img,self.gt = self.detector_augment.RandomFlip(self.img,self.gt)
        self.img = self.transform(self.img)
        return self.img, self.label, self.gt


    def __len__(self):
        return len(self.dataset)


