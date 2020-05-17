import torch.nn as nn

#Task Type
Module_Type = "CLS_BOX"       #CLS_BOX or CLS  or BOX
Data_Type = "Global"     #Global or Local
attention=True
attention_init = False
show = False

#model settings
model = dict(
    type = "ResNet"
)

#model training and testing settings
train_cfg = dict(
    criterions = {
        "CLS_BOX": [nn.CrossEntropyLoss(), nn.L1Loss()],
        "CLS": nn.CrossEntropyLoss(),
        "BOX": nn.L1Loss()
    }
)
# test_cfg = dict()

#dataset settings
dataset_type = "CLS_BOX"
data_root = "D:/Project_Glaucoma/dataset/"
img_norm_cfg = dict(
    mean=0.28604059698879553, std=0.35302424451492237, to_rgb=True)

# Mean_Box = [0.1667, 0.4459, 0.14209, 0.1531]
Mean_Box = [0.5, 0.5, 0.2, 0.2]
total_epochs = 300
batch_size = 32
lr = 1e-4
checkpoint_interval = 2

workdirs = "D:/Project_Glaucoma/DENet/weights/" + Module_Type
checkpoint = workdirs + "/epoch_251.pth"
# retrain_from = workdirs + "/epoch_99.pth"
retrain_from = None

c = checkpoint.split("/")[-1].split('.')[0]

data = dict(
    train = dict(
        type = dataset_type,
        ann_file = data_root + "JSONS/LAG_train.json",
        img_prefix = data_root+"LAG"
    ),
    eval = dict(
        type = dataset_type,
        ann_file = data_root + "JSONS/LAG_eval.json",
        img_prefix = data_root+"LAG"
    ),
    test = dict(
        type = dataset_type,
        ann_file=data_root + "JSONS/LAG_test.json",
        img_prefix=data_root+"LAG",

        metrics_file = "D:/毕业论文/experiment/metric_{0}.csv".format(checkpoint.split("/")[-1].split('.')[0]),    #
        np_box = data_root + "NPS/box/LAG_box_"+checkpoint.split("/")[-1].split('.')[0],   #type+dataset+epoch    默认                 为test结果
        np_label = data_root + "NPS/label/LAG_label_"+checkpoint.split("/")[-1].split('.')[0],         #
        show = False,
        with_gt = False
    )
)
