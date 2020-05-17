import torch
import torchvision
import torch.nn as nn
module = __import__("train")

from torchvision import transforms

import os
import json
import numpy as np

import config as cfg
from dataset import Data
from utils.utils import show_multi,Metrics
from models.net_builder  import Net_Builder
from train import Box_Generator

Testers = {"CLS_BOX":"Tester", "BOX":"Box_Generator", "CLS":"Classifier"}
transform=transforms.Compose([
                                # transforms.Resize(1024,1024),
                                transforms.ToTensor(),
                                transforms.Normalize((cfg.img_norm_cfg["mean"],), (cfg.img_norm_cfg["std"],))
                             ])

data_cfg = cfg.data["test"]

with open(data_cfg["ann_file"],'r') as f:
    dataset = json.load(f)

test_dataset_loc = Data(data_cfg["img_prefix"],dataset,transform,cfg.Data_Type)
test_loader_loc = torch.utils.data.DataLoader(test_dataset_loc, batch_size=cfg.batch_size, shuffle=False)

builder = Net_Builder(cfg.Module_Type)
tester = getattr(module, Testers[cfg.Module_Type])(
    net = builder.net.cuda(),
    test_loader=test_loader_loc,
    checkpoint=cfg.checkpoint
    # checkpoint=train_detector.net.state_dict(),
)


if __name__ == "__main__":
    if os.path.exists(data_cfg["np_box"]+".npy"):
        print("There exsits a result! ")
        pass
    else:
        tester.test()
        np.save(data_cfg["np_box"],tester.output["bboxs"])
        np.save(data_cfg["np_label"], tester.output["labels"])

    #compute metrics
    pred_file = "D:/Project_Glaucoma/dataset/NPS/label/LAG_label.npy"
    label_file = "D:/Project_Glaucoma/dataset/JSONS/LAG_test.json"
    metrics_file = "D:/毕业论文/experiment/metric.csv"

    metrics = Metrics(label_file=data_cfg["ann_file"], pred_file=data_cfg['np_label']+".npy")
    metrics.ROC_curve()
    metrics.output_cvs(metrics_file=data_cfg['metrics_file'])

    if data_cfg["show"]:
        show_multi(data_cfg["img_prefix"],data_cfg["ann_file"], data_cfg["np_box"], with_gt=data_cfg["with_gt"])






