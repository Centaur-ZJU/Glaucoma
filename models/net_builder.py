import torchvision
import torch.nn as nn
import config as cfg
from models.attention_resnet import Faster_Attention, resnet50,resnet18,BasicBlock


class Net_Builder():
    def __init__(self, Module_Type):
        self.Module_Name = Module_Type
        self.net = None

        if Module_Type == "CLS_BOX":
            self.cls_box()
        elif Module_Type == "CLS":
            self.cls()
        elif Module_Type == "BOX":
            self.box()

    def cls_box(self):
        # self.net = LC_Net()
        # self.net = resnet50(pretrained=True, num_classes=2,attention=cfg.attention,show=cfg.show)
        self.net = Faster_Attention(BasicBlock, [2, 2, 2, 2], attention=True,show=cfg.show,attention_init=cfg.attention_init)

    def cls(self):
        self.net = torchvision.models.vgg19(pretrained=True)
        self.net.classifier._modules['6'] = nn.Linear(4096, 2)

    def box(self):
        self.net = resnet18(pretrained=True)

class Fast_Net(nn.Module):
    def __init__(self):
        super(Fast_Net, self).__init__()
        self.net = torchvision.models.resnet50(prtrained=True)
        # self.features =

    def hook(self,module,input,output):
        self.features = self.features.expand(output.data.shape)
        self.features.copy_(output.data)



