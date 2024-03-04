import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, alp=1.0, prob=0.0):
        super(EEGNet, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.bn1 = nn.BatchNorm2d(16, affine=True, track_running_stats=True)
        #self.activation0 = nn.ELU(alpha=alp)
        #self.avgPool0 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Layer 2
        self.depthwiseConv2 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32, affine=True, track_running_stats=True)
        self.activation1 = nn.ELU(alpha=alp)
        self.avgPool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Layer 3
        self.separableConv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), bias=False, padding=(0, 7))
        self.bn3 = nn.BatchNorm2d(32, affine=True, track_running_stats=True)
        self.activation2 = nn.ELU(alpha=alp)
        self.drpout = nn.Dropout(p=prob)
        self.avgPool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

        # classify
        self.clf = nn.Linear(in_features=736, out_features=2, bias=True)
        #self.clf = nn.Linear(in_features=160, out_features=2)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.activation0(x)
        #x = self.avgPool0(x)

        # Layer 2
        x = self.depthwiseConv2(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.avgPool1(x)
        x = self.drpout(x)

        # Layer 3
        x = self.separableConv3(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.avgPool2(x)
        x = self.drpout(x)

        # classify
        x = x.view(x.shape[0], -1)
        x = self.clf(x)

        return x
        

