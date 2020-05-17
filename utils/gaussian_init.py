import seaborn as sns
import config as cfg
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class Gaussian_map(nn.Module):
    '''
        Examle:
            Map = Gaussian_map(100)       初始化高斯图
            Map.show()
            attension_map = Map.get_map(0.2,0.5)   #高斯中心点坐标
            Map.show(attension_map)
    '''
    def __init__(self,size):
        super(Gaussian_map, self).__init__()
        self.extend_size = 2
        self.gaussian_size = (size+self.extend_size)*2
        self.map_size = size
        center_x = size+self.extend_size
        center_y = size+self.extend_size

        R = np.sqrt(center_x ** 2 + center_y ** 2)

        self.Gauss_map = np.zeros((self.gaussian_size, self.gaussian_size))

        # 初始化高斯图
        for i in range(self.gaussian_size):
            for j in range(self.gaussian_size):
                dis = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                self.Gauss_map[i, j] = np.exp(-0.5 * dis / R)

        self.Gauss_map = torch.from_numpy(self.Gauss_map).unsqueeze(0)

    def get_map(self,center_list):
        Map = []
        for center_x,center_y in center_list:
            x_min = self.map_size*(1-center_x) + self.extend_size
            x_max = x_min+self.map_size
            y_min = self.map_size*(1-center_y) + self.extend_size
            y_max = y_min+self.map_size
            Map.append(self.Gauss_map[:,int(y_min):int(y_max),int(x_min):int(x_max)])
        result = torch.cat(Map, dim=0).unsqueeze(1)
        return result.float().cuda()


    def show(self, map=None):
        if map is None:
            map = self.Gauss_map.numpy()
        else:
            map = map.cpu().detach().numpy()[0,0]
        sns.heatmap(map, cmap="rainbow")   #YlGnBu
        plt.axis('off')
        plt.show()


class Attention_Gate(nn.Module):
    def __init__(self, in_channel,stage = 1):
        super(Attention_Gate, self).__init__()

        self.batch_size = cfg.batch_size
        self.stage = stage

        self.attention = nn.Sequential(
            nn.Conv2d(in_channel, 3, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(3 ),
            nn.Conv2d(3,1,kernel_size=3,stride=1,padding=1)
        )


    def forward(self, cur_features, former_map=None):
        features = self.attention(cur_features)

        if former_map is not None:
            feature_size = features.shape[-1]
            avg_pool = nn.AdaptiveAvgPool2d(feature_size)
            former_map = avg_pool(former_map)
            features = torch.cat([features, former_map], dim=1)

            #暂时全部改成1通道
            conv = nn.Conv2d(features.shape[1],1, kernel_size=3,stride=1,padding=1).cuda()
            features = conv(features)

        return torch.sigmoid(features)


if __name__ =="__main__":
    gts = np.array([[0.2,0.5,0.2,0.2]])
    center = [[gt[0],gt[1]] for gt in gts]
    Map = Gaussian_map(100)
    # 初始化高斯图
    # Map.show(Map.Gauss_map.unsqueeze(0))

    attension_map = Map.get_map(center)  # 高斯中心点坐标

    Map.show(attension_map)

