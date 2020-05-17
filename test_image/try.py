import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
import cv2
import os
import json
import shutil
from tqdm import tqdm


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def get_bbox(dataset, file_path):
    img_names = os.listdir(file_path)
    img_paths = [os.path.join(file_path,img_path) for img_path in img_names]
    for index,img_path in enumerate(tqdm(img_paths)):
        mask = cv2.imread(img_path)

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 200, 255, 0)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[1][1]
        x, y, w, h = cv2.boundingRect(cnt)
        x+=w/2
        y+=h/2

        data = {
            "img":str(os.path.join(file_path.split("\\")[-1],img_names[index].split(".")[0]+".jpg")),
            "label": 0 if file_path.split("\\")[-1]=="Non-Glaucoma" else 1,
            "bbox":(int(x),int(y),int(w),int(h)),
            "size":mask.shape[:2]
        }
        dataset.append(data)

def get_data(dataset, root,file,label):
    root_file =  os.path.join(root,file)
    img_names = os.listdir(root_file)
    img_paths = [os.path.join(root_file, img_path) for img_path in img_names]
    for index, img_path in enumerate(tqdm(img_paths)):
        img = cv2.imread(img_path)

        data = {
            "img": str(os.path.join(file, img_names[index])),
            "label": label,
            "bbox": (),
            "size": img.shape[:2]
        }
        dataset.append(data)

def move_imgs(root):
    for person in tqdm(os.listdir(root)):
        root_person = os.path.join(root, person)
        imgs = os.listdir(root_person)

        for img in imgs:
            old_path = os.path.join(root_person, img)
            new_path = os.path.join(root, 'total', img)
            try:
                shutil.copy(old_path, new_path)
            except:
                pass

def crop_image(root,json_path, show=False):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    img_paths = [os.path.join(root,img_info["img"]) for img_info in dataset]

    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        W,H,_ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 180 , 255, 0)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        position = x/W
        if position>0.5:
            print(position)
            fliped_img = cv2.flip(img, 1)
            cv2.imwrite(img_path, fliped_img)

        # img = img[y:y+h,x:x+w,:]
        # cv2.imwrite(img_path,img)

        if show:
            # cv2.rectangle(thresh, (max(int(x), 0), max(int(y ), 0)), (int(x + w), int(y + h)), (0, 255, 0), 5)
            cv2.imshow('img',cv2.resize(thresh,(640,640)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def flip(img_path):
    img = cv2.imread(img_path)
    fliped_img = cv2.flip(img,1)

    cv2.imshow('img', cv2.resize(img, (640, 640)))
    cv2.imshow('flip', cv2.resize(fliped_img, (640, 640)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def change_size(image,read_file):
    # image = cv2.imread(read_file, 1)  # 读取图片 image_name应该是变量
    img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    print(binary_image.shape)  # 改为单通道

    x = binary_image.shape[0]
    y = binary_image.shape[1]
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right - left  # 宽度
    bottom = min(edges_y)  # 底部
    top = max(edges_y)  # 顶部
    height = top - bottom  # 高度

    pre1_picture = image[left:left + width, bottom:bottom + height]  #
    cv2.imwrite(read_file,pre1_picture)
    # cv2.imshow('img', cv2.resize(pre1_picture, (640, 640)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def multi_crop(root):
    img_names = os.listdir(root)
    img_paths = [os.path.join(root,img_name) for img_name in img_names]
    for img_path in img_paths:
        image = cv2.imread(img_path, 1)
        left_mean = image[:,0,:].mean()
        right_mean = image[:, -1, :].mean()
        if left_mean<3 or right_mean<3:
            print(left_mean,right_mean)
            change_size(image,img_path)


def merge_json_np(json_path, np_path, output_json):
    with open(json_path, 'r') as f:
        data = json.load(f)
    bboxs = np.load(np_path + ".npy")

    for index in range(len(data)):
        data[index]["bbox"] = bboxs[index]


if __name__=="__main__":
    merge_json_np(json_path = "D:/Project_Glaucoma/dataset/JSONS/data.json",
                  np_path = "D:/Project_Glaucoma/DENet/bboxs_with_anchor_199.npy",
                  output_json= "D:/Project_Glaucoma/dataset/JSONS/data_with_bbox.json")

    # img_path = "D:/Project_Glaucoma/DENet/test_image/CS30498_R.jpg"
    # mask_root = "D:/Project_Glaucoma/dataset/Annotation-Training400/Annotation-Training400/Disc_Cup_Masks"
    # negative = "Non-Glaucoma"
    # positive = "Glaucoma"
    # positive_file = os.path.join(mask_root,positive)
    # negative_file = os.path.join(mask_root,negative)
    #
    root = "D:/Project_Glaucoma/dataset/data/Zhe2_hospital"
    file1 = "normal/total"
    file2 = "Glaucoma-single/left"
    # file3 = "Glaucoma-ou/right"
    file3 = "Glaucoma-ou/left"
    dataset = []
    # get_data(dataset,root,file1,0)
    # get_data(dataset, root, file2, 1)
    # get_data(dataset, root, file3, 1)
    #
    # #
    # # for data in dataset:
    # #     print(data["size"])
        # with open(os.path.join("D:/Project_Glaucoma/dataset/JSONS/data.json"),"w") as f:
        #     json.dump(dataset,f)
    #
    #
    # root = "D:/Project_Glaucoma/dataset/data/Zhe2_hospital/normal"
    # move_imgs(root)
    # crop_image(root,"D:/Project_Glaucoma/dataset/JSONS/data.json",show=True)
    #20190102112826_31000901_58862.jpg 20190102164824_31000901_49232.jpg
    # multi_crop("D:/Project_Glaucoma/dataset/data/Zhe2_hospital/normal/total")