import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class double_conv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class U_Net(nn.Module):

    def __init__(self):
        super(U_Net,self).__init__()

        self.down1 = double_conv(3,32)

        self.pool2 = nn.MaxPool2d(2)
        self.down2 = double_conv(32,64)

        self.pool3 = nn.MaxPool2d(2)
        self.down3 = double_conv(64, 128)

        self.pool4 = nn.MaxPool2d(2)
        self.down4 = double_conv(128, 256)

        self.pool5 = nn.MaxPool2d(2)
        self.down5 = double_conv(256, 512)

        #Then we upsample the network
        self.deconv4 = nn.ConvTranspose2d(512,512, kernel_size=2, stride=2)
        self.up4 = double_conv(256+512, 256)

        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up3 = double_conv(128 + 256, 128)

        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up2 = double_conv(64 + 128, 64)

        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up1 = double_conv(32 + 64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        down1 = self.down1(x)

        down2 = self.pool2(down1)
        down2 = self.down2(down2)

        down3 = self.pool3(down2)
        down3= self.down3(down3)

        down4 = self.pool4(down3)
        down4 = self.down4(down4)

        down5 = self.pool5(down4)
        down5 = self.down5(down5)

        up4 = torch.cat([down4,self.deconv4(down5)],dim=1)
        up4 = self.up4(up4)

        up3 = torch.cat([down3, self.deconv3(up4)], dim=1)
        up3 = self.up3(up3)

        up2 = torch.cat([down2, self.deconv2(up3)], dim=1)
        up2 = self.up2(up2)

        up1 = torch.cat([down1, self.deconv1(up2)], dim=1)
        up1 = self.up1(up1)

        out = self.out(up1)

        return out

class Left_UNet(nn.Module):

    def __init__(self):
        super(Left_UNet,self).__init__()

        self.down1 = double_conv(3,32)

        self.pool2 = nn.MaxPool2d(2)
        self.down2 = double_conv(32,64)

        self.pool3 = nn.MaxPool2d(2)
        self.down3 = double_conv(64, 128)

        self.pool4 = nn.MaxPool2d(2)
        self.down4 = double_conv(128, 256)

        self.pool5 = nn.MaxPool2d(2)
        self.down5 = double_conv(256, 512)

        self.average_pool = nn.AvgPool2d(7)

        self.out = nn.Sequential(
            nn.Linear(12800, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        down1 = self.down1(x)

        down2 = self.pool2(down1)
        down2 = self.down2(down2)

        down3 = self.pool3(down2)
        down3 = self.down3(down3)

        down4 = self.pool4(down3)
        down4 = self.down4(down4)

        down5 = self.pool5(down4)
        down5 = self.down5(down5)

        x = self.average_pool(down5)
        x = x.reshape(x.shape[0],-1)
        x = self.out(x)
        x = F.softmax(x)
        return x

if __name__ == "__main__":
    X = torch.randn((1,3,640,640))
    model = Left_UNet()
    Y = model(X)
    print(Y)