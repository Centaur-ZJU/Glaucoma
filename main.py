import os
import json
import numpy as np
module = __import__("train")

import torch
from torchvision import transforms

import config as cfg
from dataset import Data
from utils.utils import show_box
from models.net_builder import Net_Builder


train_cfg = cfg.data["train"]
eval_cfg = cfg.data["eval"]
Trainers = {"CLS_BOX":"Trainer", "BOX":"Detector", "CLS":"Classifier"}

with open(train_cfg["ann_file"],'r') as f:
    t_dataset = json.load(f)
with open(eval_cfg["ann_file"], 'r') as f:
    e_dataset = json.load(f)

transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((cfg.img_norm_cfg["mean"],), (cfg.img_norm_cfg["std"],))
                             ])

train_dataset = Data(train_cfg["img_prefix"],t_dataset,transform, Module_Type=cfg.Module_Type)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
eval_dataset = Data(eval_cfg["img_prefix"],e_dataset,transform, Module_Type=cfg.Module_Type)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

builder = Net_Builder(cfg.Module_Type)
trainer = getattr(module, Trainers[cfg.Module_Type])(
        net=builder.net.cuda(),
        criterions=cfg.train_cfg["criterions"][cfg.Module_Type],
        train_loader=train_loader,
        test_loader=eval_loader,
        lr=cfg.lr,
        resume_point=cfg.retrain_from
    )

# weight = torch.from_numpy(np.array([0.1,1.0])).float()

def train():
    now_epoch = 0
    if cfg.retrain_from is not None:
        now_epoch = int(cfg.retrain_from.split("epoch_")[-1].split(".")[0])

    for epoch in range(now_epoch,cfg.total_epochs):
        print("Epoch %d training....."%(epoch))
        trainer.train(epoch)
        print("Epoch %d testing....."%(epoch))
        trainer.eval(epoch)

        if (epoch+1) % cfg.checkpoint_interval == 0:
            torch.save(trainer.net.state_dict(),cfg.workdirs+"/epoch_%d.pth"%(epoch))


if __name__ == "__main__":
    train()
    # root = cfg.data["test"]["img_prefix"]
    # json_path = "D:/Project_Glaucoma/dataset/JSONS/normal.json"
    # with open(json_path,'r') as f:
    #     infos = json.load(f)
    # for info in infos:
    #     W,H = info["size"]
    #     info['bbox'] = [info["bbox"][0]/W,info["bbox"][1]/H, info["bbox"][2]/W, info["bbox"][3]/H]
    #     img_name = root+"/"+info["img"]
    #
    #     from PIL import Image
    #
    #     if info['bbox'][1]>0.5:
    #         print(img_name)
    #         img = Image.open(img_name)
    #         out = img.transpose(Image.FLIP_LEFT_RIGHT)
            # out.save(img_name)


        # show_box(img,[np.array(bbox)])

    # with open("D:/Project_Glaucoma/dataset/JSONS/normal.json", 'w') as f:
    #     json.dump(infos, f)






