from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from utils.utils import Encode, Decode,compute_IOU,show_box
import config as cfg


class Trainer():
    def __init__(self,net,criterions,train_loader,test_loader, lr, resume_point=None):
        self.net = net
        self.criterions = criterions
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.resume_point = resume_point

        self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 8, gamma=0.1, last_epoch=-1)

        if self.resume_point is not None:
            self.net.load_state_dict(torch.load(self.resume_point))

        self.Mean_Box = Variable(torch.Tensor(cfg.Mean_Box),requires_grad=False)
        self.total_loss = 0
        self.writer = SummaryWriter()

        self.scalars_log = {}
        self.output = {"labels":[],
                       "bboxs":[]}

    def train(self, epoch):
        self.net.train()
        True_Predict = 0
        for i, data in enumerate(tqdm(self.train_loader)):
            img, label, gt = data
            img, label, gt = img.cuda(), label.cuda(), gt.float().cuda()
            self.optimizer.zero_grad()

            outs = self.net(img,gt)    #cls, box_reg
            self.compute_forward(outs,(label,gt),i + epoch * len(self.train_loader))

            self.loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outs[0], 1)
            # print(predicted,label)
            True_Predict += (predicted == label).sum().item()

        acc = 1.0 * True_Predict / (len(self.train_loader)*cfg.batch_size)
        self.writer.add_scalar('train/acc', acc, epoch)



    def compute_forward(self,output, targets, index):
        """
        :param targets: (label,gt)
        :return:
        """
        gt, d_box = targets[1], self.Mean_Box.cuda()
        l_box, d_box = output[1], self.Mean_Box.cuda()
        l_gt, b_box = Encode(d_box, gt), Decode(d_box, l_box)

        loss_cls = self.criterions[0](output[0],targets[0])
        loss_box = self.criterions[1](l_box, l_gt)
        self.loss = loss_cls + loss_box

        iou_mean = compute_IOU(b_box, gt).sum() / cfg.batch_size

        self.writer.add_scalars("train/loss", {"loss_cls": loss_cls.item(),
                                              "loss_box": loss_box.item(),
                                              "loss": self.loss.item()},index)
        self.writer.add_scalar('train/IOU', iou_mean, index)

    def eval(self,epoch):
        self.net.eval()
        True_Predict = 0
        for i, data in enumerate(tqdm(self.test_loader)):
            img, label, gt = data
            img, label, gt= img.cuda(), label.cuda(), gt.float().cuda()

            outs = self.net(img,gt)
            l_box, d_box = outs[1], self.Mean_Box.cuda()
            l_gt, b_box = Encode(d_box, gt), Decode(d_box, l_box)
            iou_mean = compute_IOU(b_box, gt).sum()

            # bboxs = [out.cpu().detach().numpy() for out in outs]
            bboxs = [out.cpu().detach().numpy() for out in b_box]
            self.output["bboxs"] += bboxs


            _, predicted = torch.max(outs[0], 1)
            # print(predicted,label)
            True_Predict += (predicted == label).sum().item()
            self.writer.add_scalar('eval/IOU', iou_mean, i + epoch * len(self.test_loader))

        acc = 1.0 * True_Predict / (len(self.test_loader))
        self.writer.add_scalar('eval/acc', acc, epoch)


class Tester():
    def __init__(self,net,test_loader,checkpoint=None):
        self.net = net
        self.test_loader = test_loader
        self.checkpoint = checkpoint
        self.Mean_Box = Variable(torch.Tensor(cfg.Mean_Box), requires_grad=False)

        self.output = {"labels": [],
                       "bboxs": []}

        if self.checkpoint is not None:
            if isinstance(self.checkpoint,str):
                self.net.load_state_dict(torch.load(self.checkpoint))
            else:
                self.net.load_state_dict(self.checkpoint)

        # self.writer = SummaryWriter()
        self.net.eval()

    def regist_result(self,results):
        bboxs = [bbox.cpu().detach().numpy() for bbox in results[1]]
        self.output["bboxs"] += bboxs

        # _, labels = torch.max(results[0], 1)
        labels = F.softmax(results[0], dim=1)
        labels = [label[0].cpu().detach().numpy() for label in labels]
        self.output["labels"] += labels

    def test(self):
        True_Predict = 0

        for i, data in enumerate(tqdm(self.test_loader)):
            img, label, gt = data
            img, label, gt = img.cuda(), label.cuda(), gt.float().cuda()

            # outs = self.net(img)
            outs = self.net(img, gt)  # cls, box_reg
            l_box, d_box = outs[1], self.Mean_Box.cuda()
            l_gt, b_box = Encode(d_box, gt), Decode(d_box, l_box)
            self.regist_result([outs[0],b_box])

            _, predicted = torch.max(outs[0], 1)

            True_Predict += (predicted == label).sum().item()
            # print("loss: %.2f"%(loss.item()))
        # self.writer.add_scalar('test/acc', 1.0 * True_Predict / (len(self.test_loader)*cfg.batch_size), )


class Classifier():
    def __init__(self, net, criterions, train_loader,test_loader, lr,resume_point=None):
        self.net = net
        self.criterions = criterions
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 8, gamma=0.1, last_epoch=-1)
        self.resume_point = resume_point

        if self.resume_point is not None:
            self.net.load_state_dict(torch.load(self.resume_point))

        self.total_loss = 0
        self.writer = SummaryWriter()

        self.scalars_log = {}
        self.output = {"labels": [],
                           "bboxs": []}

    def train(self, epoch):
        True_Predict = 0
        self.net.train()
        for i, data in enumerate(tqdm(self.train_loader)):
            img, label, gt = data
            img, label, gt = img.cuda(), label.cuda(), gt.float().cuda()
            self.optimizer.zero_grad()

            outs = self.net(img)
            self.loss = self.criterions(outs, label)
            self.total_loss += self.loss.item()

            self.loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outs, 1)
            True_Predict += (predicted == label).sum().item()

            self.writer.add_scalar('train/cls/loss', self.loss.item(),  i + epoch * len(self.train_loader))

        acc = 1.0 * True_Predict / (len(self.train_loader) * cfg.batch_size)
        self.writer.add_scalar('train/cls/acc', acc, epoch)


    def test(self,epoch):
        True_Predict = 0
        self.net.eval()
        for i, data in enumerate(tqdm(self.train_loader)):
            img, label, gt = data
            img, label, gt = img.cuda(), label.cuda(), gt.float().cuda()

            outs = self.net(img)
            _, predicted = torch.max(outs, 1)
            True_Predict += (predicted == label).sum().item()

        acc = 1.0 * True_Predict / (len(self.train_loader) * len(outs))
        self.writer.add_scalar('test/cls/acc', acc, epoch)

class Detector():
    def __init__(self, net, criterions, train_loader,test_loader, lr,resume_point=None):
        self.net = net
        self.criterions = criterions
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 8, gamma=0.1, last_epoch=-1)
        self.resume_point = resume_point

        if self.resume_point is not None:
            self.net.load_state_dict(torch.load(self.resume_point))


        self.Mean_Box = Variable(torch.Tensor(cfg.Mean_Box), requires_grad=False)
        self.total_loss = 0
        self.writer = SummaryWriter()

        self.scalars_log = {}
        self.output = {"labels": [],
                           "bboxs": []}

    def train(self, epoch):
        True_Predict = 0
        # self.net.train()
        for i, data in enumerate(tqdm(self.train_loader)):
            img, label, gt = data
            img, label, gt = img.cuda(), label.cuda(), gt.float().cuda()
            self.optimizer.zero_grad()

            outs = self.net(img)
            self.compute_forward(outs, (label, gt), i + epoch * len(self.train_loader))

            self.loss.backward()
            self.optimizer.step()

    def compute_forward(self, l_box, targets, index):
        """
        :param targets: (label,gt)
        :return:
        """
        gt,d_box = targets[1],self.Mean_Box.cuda()
        l_gt,b_box = Encode(d_box,gt), Decode(d_box,l_box)
        self.loss = self.criterions(l_box, l_gt)
        iou_mean = compute_IOU(b_box, gt).sum() / cfg.batch_size
        # bbox = Decode(self.Mean_Box.cuda(),output)
        # self.loss = self.criterions(bbox, targets[1])
        # iou_mean = compute_IOU(bbox, targets[1]).sum() / len(output)

        # print(iou_mean)
        self.writer.add_scalar('train/loc/IOU', iou_mean, index)
        self.writer.add_scalar('train/loc/loss', self.loss.item(), index)

    def test(self,epoch):
        self.net.eval()
        for i, data in enumerate(tqdm(self.test_loader)):
            img, label, gt = data
            img, label, gt = img.cuda(), label.cuda(), gt.float().cuda()

            outs = self.net(img)
            l_box, d_box = outs, self.Mean_Box.cuda()
            l_gt, b_box = Encode(d_box, gt), Decode(d_box, l_box)
            iou_mean = compute_IOU(b_box, gt).sum() / cfg.batch_size
            # bbox = Decode(self.Mean_Box.cuda(), outs)
            # iou_mean = compute_IOU(bbox, gt).sum() / len(outs)

            # bboxs = [out.cpu().detach().numpy() for out in outs]
            bboxs = [out.cpu().detach().numpy() for out in b_box]
            self.output["bboxs"] += bboxs

            self.writer.add_scalar('train/test/IOU', iou_mean, i + epoch * len(self.test_loader))


class Box_Generator():
    def __init__(self, net,test_loader, checkpoint=None):
        self.net = net
        self.test_loader = test_loader
        self.checkpoint = checkpoint


        if self.checkpoint is not None:
            if isinstance(self.checkpoint,str):
                self.net.load_state_dict(torch.load(self.checkpoint))
            else:
                self.net.load_state_dict(self.checkpoint)
        self.net.eval()

        self.Mean_Box = Variable(torch.Tensor(cfg.Mean_Box), requires_grad=False)
        self.writer = SummaryWriter()

        self.scalars_log = {}
        self.output = {"labels": [],
                           "bboxs": []}

    def test(self):
        for i, data in enumerate(tqdm(self.test_loader)):
            img, label, gt = data
            img, label, gt = img.cuda(), label.cuda(), gt.float().cuda()

            outs = self.net(img)
            # bbox = outs
            # bbox = Decode(self.Mean_Box.cuda(), outs)
            # iou_mean = compute_IOU(bbox, gt).sum() / len(outs)
            l_box,d_box =outs, self.Mean_Box.cuda()
            l_gt, b_box = Encode(d_box, gt), Decode(d_box, l_box)

            iou_mean = compute_IOU(b_box, gt).sum() / cfg.batch_size

            # bboxs = [out.cpu().detach().numpy() for out in outs]
            bboxs = [out.cpu().detach().numpy() for out in b_box]
            self.output["bboxs"] += bboxs

            self.writer.add_scalar('test/IOU', iou_mean, i)



if __name__ == "__main__":
    x = Variable(torch.randn([1, 3, 1024, 1024]))

    # y1 = F.interpolate(x, size=[32, 32])

    # y2 = F.interpolate(x, size=[224, 224], mode="bilinear")
    # for epoch in range(n_epochs):
    #
    #     if (epoch+1) % 10 == 0:
    #         torch.save(location_net.state_dict(),"weights/epoch_%d.pth"%(epoch))






