import numpy as np
import scipy
import torch
import json
import os
import pandas
from skimage.measure import label, regionprops
from  sklearn.metrics import confusion_matrix
from skimage.transform import rotate, resize

import cv2


def pro_process(temp_img, input_size):
    img = np.asarray(temp_img).astype('float32')
    img = scipy.misc.imresize(img, (input_size, input_size, 3))
    return img


def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0

    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def Deep_Screening(target_model, tmp_img, input_size):
    temp_img = scipy.misc.imresize(tmp_img, (input_size, input_size, 3))
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    Pre_result = target_model.predict(temp_img)
    return Pre_result


def Disc_Crop(org_img, DiscROI_size, C_x, C_y):
    disc_region = np.zeros((DiscROI_size, DiscROI_size, 3), dtype=org_img.dtype)
    crop_coord = [int(C_x - DiscROI_size / 2), int(C_x + DiscROI_size / 2), int(C_y - DiscROI_size / 2),
                  int(C_y + DiscROI_size / 2)]
    err_coord = [0, DiscROI_size, 0, DiscROI_size]

    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0]) + 1
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2]) + 1
        crop_coord[2] = 0

    if crop_coord[1] > org_img.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - org_img.shape[0]) - 1
        crop_coord[1] = org_img.shape[0]

    if crop_coord[3] > org_img.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - org_img.shape[1]) - 1
        crop_coord[3] = org_img.shape[1]

    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[crop_coord[0]:crop_coord[1],
                                                                          crop_coord[2]:crop_coord[3], ]

    return disc_region

def compute_IOU(box_a,box_b):
    box_a = box_a.cpu().detach().numpy()
    box_b = box_b.cpu().detach().numpy()

    xa, ya, wa, ha = box_a[:,0],box_a[:,1],box_a[:,2],box_a[:,3]
    xb, yb, wb, hb = box_b[:,0],box_b[:,1],box_b[:,2],box_b[:,3]
    xa_1,ya_1,xa_2,ya_2 = xa - wa/2, ya - ha/2, xa + wa/2, ya + ha/2
    xb_1, yb_1, xb_2, yb_2 = xb - wb / 2, yb - hb / 2, xb + wb / 2, yb + hb / 2

    left_line = np.maximum(xa_1,xb_1)
    right_line = np.minimum(xa_2,xb_2)
    top_line = np.maximum(ya_1,yb_1)
    bottom_line = np.minimum(ya_2,yb_2)

    sum_area = wa*ha + wb*hb


    intersected = ((left_line >= right_line) + (top_line >= bottom_line))==0

    intersect = (right_line - left_line) * (bottom_line - top_line)

    iou = (intersect / (sum_area - intersect))*1.0 * intersected
    return iou


def Decode(d_box, l_box):
    d_box = d_box.expand(l_box.shape)
    dx, dy, dw, dh = d_box[:, 0], d_box[:, 1], d_box[:, 2], d_box[:, 3]
    lx, ly, lw, lh = l_box[:, 0], l_box[:, 1], l_box[:, 2], l_box[:, 3]

    x = (dw * lx + dx).view(-1,1)
    y = (dh * ly + dy).view(-1,1)
    w = (dw * torch.exp(lw)).view(-1,1)
    h = (dh * torch.exp(lh)).view(-1,1)

    result = torch.cat((x,y,w,h),1)
    return result

def Encode(d_box,b_box):
    d_box = d_box.expand(b_box.shape)
    dx, dy, dw, dh = d_box[:, 0], d_box[:, 1], d_box[:, 2], d_box[:, 3]
    bx, by, bw, bh = b_box[:, 0], b_box[:, 1], b_box[:, 2], b_box[:, 3]

    x = (1.0*(bx-dx)/dw).view(-1,1)
    y = (1.0*(by-dy)/dh).view(-1,1)
    w = torch.log(bw/dw).view(-1,1)
    h = torch.log(bh/dh).view(-1,1)

    return torch.cat((x,y,w,h),1)


def show_box(img, bboxs,factor = 0,origin=False):
    """
    :param img: img or path, need to judge
    :param bboxs: list of [x,y,w,h]
    :param is_origin: 0~255:True, 0.0~1.0:False(need to transfer)
    :return: None
    """
    if isinstance(img,str):
        img = cv2.imread(img)
    img = cv2.resize(img,(640,640))
    W,H,_= img.shape

    for index,bbox in enumerate(bboxs):
        x,y,w,h = bbox*W
        cv2.rectangle(img, (max(int(x-w/2),0),max(int(y-h/2),0)), (int(x+w/2), int(y+h/2)),(0,255*index,255*(index+1)),2)
        if factor is not 0:
            w, h = w*factor,h*factor
            cv2.rectangle(img, (max(int(x - w / 2), 0), max(int(y - h / 2), 0)), (int(x + w / 2), int(y + h / 2)),(0, 255 * index, 255 * (index + 1)), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_multi(root,json_path, np_path,with_gt=False):
    """
    :param json_path: json of img infos
    :param np_path: np of boxs
    :return:None
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    img_names = [img_name["img"] for img_name in data]
    bboxs = np.load(np_path+".npy")

    if with_gt:
        origin_bboxs = []
        for img in data:
            w,h = img["size"]
            bbox = img["bbox"]
            origin_bboxs.append(np.array([bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]))

        for img_name,bbox,origin_bbox in zip(img_names,bboxs,origin_bboxs):
            show_box(os.path.join(root,img_name),[bbox,origin_bbox])
    else:
        for img_name, bbox in zip(img_names, bboxs):
            print(bbox)
            show_box(os.path.join(root, img_name), [bbox])

#normal 0, glaucoma 1
class Metrics:
    def __init__(self,label_file,pred_file):
        self.preds = np.load(pred_file)
        with open(label_file, 'r') as f:
            data = json.load(f)
            self.labels = [label["label"] for label in data]

        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0

        self.Sensitivitys = []
        self.Specificitys = []
        self.thresholds = []
        self.TPRs = []
        self.FPRs = []
        self.F_scores = []

    def compute_metrics(self,threshold):
        predicts = self.preds<threshold
        labels = self.labels
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0

        if threshold==0.5:
            acc = 1.0*(labels==predicts).sum()/len(labels)
            print("Accuracy: {0}".format(acc))

        matrix = confusion_matrix(labels,predicts)
        self.TP,self.FN = [float(a) for a in matrix[0]]
        self.FP,self.TN = [float(a) for a in matrix[1]]

    def Sensitivity_Specificity(self):
        self.Sensitivity = 1.0 * self.TP / (self.TP + self.FN)
        self.Specificity = 1.0 * self.TN / (self.TN + self.FP)

    def F_score(self,beta=2):
        beta_square = beta * beta
        self.F2_score = 1.0 * (1 + beta_square) * self.TP / ((1 + beta_square) * self.TP + beta_square * self.FN + self.FP)

    def TPR_FPR(self):
        TPR = 1.0 * self.TP / (self.TP + self.FN)
        FPR = 1.0 * self.FP / (self.FP + self.TN)
        return TPR,FPR

    def ROC_curve(self,space = 0.1):
        threshold = 0.0
        while threshold<=1.0:
            self.compute_metrics(threshold = threshold)
            self.Sensitivity_Specificity()
            self.F_score(beta=2)
            TPR,FPR = self.TPR_FPR()
            if threshold==0.5:
                print("Sensitivity: {2}\nSpecificity: {1}\nF2_score: {0}".format(self.F2_score,self.Specificity,self.Sensitivity))


            self.thresholds.append(threshold)
            self.Sensitivitys.append(self.Sensitivity)
            self.Specificitys.append(self.Specificity)
            self.TPRs.append(TPR)
            self.FPRs.append(FPR)
            self.F_scores.append(self.F2_score)

            threshold+=space

    def output_cvs(self,metrics_file):
        result = []
        for threshold, Sensitivity, Specificity, TPR, FPR, F_score in zip(self.thresholds, self.Sensitivitys, self.Specificitys, self.TPRs,
                                                                          self.FPRs, self.F_scores):
            result.append([threshold, Sensitivity, Specificity, TPR, FPR, F_score])

        column_name = ['Threshold', 'Sensitivitys', 'Specificitys', 'TPRs', 'FPRs', 'F_scores']
        xml_df = pandas.DataFrame(result, columns=column_name)

        xml_df.to_csv(metrics_file, index=None)

if __name__ == "__main__":
    # show_multi(json_path = "D:/Project_Glaucoma/dataset/data.json",np_path="D:/Project_Glaucoma/DENet/bboxs_with_anchor_9_9_13.npy")

    pass

    # json_path = "D:/Project_Glaucoma/dataset/bbox_label.json"
    # root = "D:/Project_Glaucoma/dataset/REFUGE-Training400/Training400"
    # with open(json_path, 'r') as f:
    #     dataset = json.load(f)
    #
    # for data in dataset[:45]:
    #     print(data,'\n')


    # root = "D:/Project_Glaucoma/dataset/REFUGE-Training400/Training400"

    # with open("D:/Project_Glaucoma/dataset/bbox_label.json", 'r') as f:
    #     dataset = json.load(f)
    # for data in dataset:
    #     show_box(os.path.join(root,data["img"]),[data["bbox"]])

    # img_path = "D:/Project_Glaucoma/DENet/test_image/CS30498_R.jpg"
    # img = cv2.imread(img_path)
    # rect1 = torch.randn((2,4))
    # # (top, left, bottom, right)
    # rect2 = torch.randn((2,4))
    # iou = compute_IOU(rect1, rect2)
    # print(iou)

    # print(img.shape)
    # disc_region = Disc_Crop(img,850,1230,1875)
    # w,h,_ = disc_region.shape
    #
    # linear_polar = cv2.linearPolar(disc_region, (w / 2, w / 2), w/3, cv2.WARP_FILL_OUTLIERS)
    # 1540,900
    # 2220,1550
    # 680  650
    # cv2.imshow('disc_region', disc_region)
    # cv2.imshow('Linear_Polar', linear_polar)


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # from PIL import Image
    # from PIL import ImageEnhance
    #
    # # 原始图像
    # image = Image.open(img_path)
    # image.save('origin.jpg')
    #
    # # 对比度增强
    # enh_con = ImageEnhance.Contrast(image)
    # contrast = 1.5
    # image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.save("contrast.jpg")
    #
    # # 锐度增强
    # enh_sha = ImageEnhance.Sharpness(image)
    # sharpness = 10
    # image_sharped = enh_sha.enhance(sharpness)
    # image_sharped.save("sharpness.jpg")
