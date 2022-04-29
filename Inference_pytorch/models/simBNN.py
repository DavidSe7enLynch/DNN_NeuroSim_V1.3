from re import X
from utee import misc
import torch.nn as nn
import torch.nn.functional as F 
import torch

eps = 0.00002 
bn_mom = 0.9
affine = True 

class FastSign(nn.Module):
    def __init__(self):
        super(FastSign, self).__init__()

    def forward(self, input):
        out_forward = torch.sign(input)
        ''' 
        Only inputs in the range [-t_clip,t_clip] 
        have gradient 1. 
        '''
        t_clip = 1.3
        out_backward = torch.clamp(input, -t_clip, t_clip)
        return (out_forward.detach() 
                - out_backward.detach() + out_backward)

class BinaryConv2d(nn.Conv2d):
    '''
    A convolutional layer with its weight tensor binarized to {-1, +1}.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=(0,0), dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(BinaryConv2d, self).__init__(in_channels, out_channels,
                                              kernel_size, stride,
                                              padding, dilation, groups,
                                              bias, padding_mode)
        self.binarize = FastSign()

    def forward(self, input):
        return F.conv2d(self.binarize(input), self.binarize(self.weight),
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

class BinaryFully_connected(nn.linear):
    def __init__(self, in_features, out_features, bias = True, device = None, dtype = None):
        super(BinaryFully_connected, self).__init__(in_features, out_features, bias, device, dtype)
        
        self.binarize = FastSign()

    def forward(self, input):
        return F.linear(self.binarize(input), self.binarize(self.weight), self.bias, self.device, self.dtype)

class simBNN(nn.Module):
    def __init__(self, num_classes):
        super(simBNN, self).__init__()

        # Stage 1 
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 96 ,kernel_size = (11, 11),stride = (4, 4)) 
        # In_channel would change based on the data structure 
        self.bn1 = nn.BatchNorm2d(96, eps = eps, affine = affine, momentum = bn_mom)

        # Stage 2 
        self.act_q2 = FastSign()
        self.conv2 = BinaryConv2d(in_channels = 96, out_channels = 256 ,kernel_size = (5, 5), padding = (2,2)) 
        # act_bit=BITA & weight_bit=BITW ? 
        self.bn2 = nn.BatchNorm2d(256, eps = eps, affine = affine, momentum = bn_mom)

        # Stage 3 
        self.act_q3 = FastSign()
        self.conv3 = BinaryConv2d(in_channels = 256, out_channels = 384 ,kernel_size = (3, 3), padding = (1,1))
        self.bn3 = nn.BatchNorm2d(384, eps = eps, affine = affine, momentum = bn_mom)

        self.act_q4 = FastSign()
        self.conv4 = BinaryConv2d(in_channels = 384, out_channels = 384 ,kernel_size = (3, 3), padding = (1,1))
        self.bn4 = nn.BatchNorm2d(384, eps = eps, affine = affine, momentum = bn_mom)

        self.act_q5 = FastSign()
        self.conv5 = BinaryConv2d(in_channels = 384, out_channels = 256 ,kernel_size = (3, 3), padding = (1,1))
        self.bn5 = nn.BatchNorm2d(256, eps = eps, affine = affine, momentum = bn_mom)

        # Stage 4
        self.act_fc1 = FastSign()
        self.fc1 = BinaryFully_connected(in_features =256, out_features=4096 )
        self.bn6 = nn.BatchNorm2d(4096, eps = eps, affine = affine, momentum = bn_mom)

        # Stage 5
        self.act_fc2 = FastSign()
        self.fc2 = BinaryFully_connected(in_features = 4096, out_features=4096 ) 
        self.bn7 = nn.BatchNorm2d(4096, eps = eps, affine = affine, momentum = bn_mom)

        # Stage 6 
        self.fc3 = nn.linear(in_features = 4096, out_features =num_classes )
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        # Stage 1 
        x=self.conv1(x)
        x=F.relu(self.bn1(x))
        x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2))

        # Stage 2 
        x=self.act_q2(x) 
        x=self.conv2(x) 
        x=self.bn2(x) 
        x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2))

        #Stage 3 
        x=self.act_q3(x) 
        x=self.conv3(x) 
        x=self.bn3(x) 
        x=self.act_q4(X) 
        x=self.conv4(x) 
        x=self.bn4(x) 
        x=self.act_q5(x) 
        x=self.conv5(x) 
        x=self.bn5(x) 
        x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2))

        #Stage 4 
        x=self.act_fc1() 
        x = self.fc1(x)
        x=F.relu(self.bn6(x))

        #Stage 5 
        x=self.act_fc2() 
        x = self.fc2(x) 
        x=F.relu(self.bn7(x))

        #Stage 6 
        x = self.fc3(x)
        x = self.softmax (x)

def simBNN():
    model = simBNN()
    return model