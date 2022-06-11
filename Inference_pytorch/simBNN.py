import argparse
import os
import time
from re import X
from utee import misc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch
from models import dataset
import torch.optim as optim
from torch.autograd import Variable
from utee import wage_util

# Assume the input is Nx3x32x32

# Global Variables 
in_channels = 3 
batch_size = 64
lr=0.001
epoch_num = 2 
seed = 117 

eps = 0.00002 
bn_mom = 0.9
affine = True 

# Load Data Set (Nx3x256x256)
train_loader, test_loader = dataset.get_imagenet(batch_size=batch_size, num_workers=1)
#train_loader, test_loader = dataset.get_cifar10(batch_size=batch_size, num_workers=1)

# Define the BNN model 
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

class BinaryFully_connected(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        super(BinaryFully_connected, self).__init__(in_features, out_features, bias)
        
        self.binarize = FastSign()

    def forward(self, input):
        return F.linear(self.binarize(input), self.binarize(self.weight)) #, self.bias, self.device, self.dtype)

class simBNN(nn.Module):
    def __init__(self, num_classes):
        super(simBNN, self).__init__()

        # Stage 1 
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 96 ,kernel_size = (11, 11),stride = (4, 4))
        # Now size is Nx96x62x62
        self.bn1 = nn.BatchNorm2d(96, eps = eps, affine = affine, momentum = bn_mom)
        # Now size is Nx96x62x62
        # After max_pool, size is  Nx96x30x30

        # Stage 2 
        self.act_q2 = FastSign()
        self.conv2 = BinaryConv2d(in_channels = 96, out_channels = 256 ,kernel_size = (5, 5), padding = (2,2)) 
        # Now size is Nx256x15x15
        self.bn2 = nn.BatchNorm2d(256, eps = eps, affine = affine, momentum = bn_mom)
        # After max_pool, size is Nx256x7x7

        # Stage 3 
        self.act_q3 = FastSign()
        self.conv3 = BinaryConv2d(in_channels = 256, out_channels = 384 ,kernel_size = (3, 3), padding = (1,1))
        # Now size is Nx384x7x7 
        self.bn3 = nn.BatchNorm2d(384, eps = eps, affine = affine, momentum = bn_mom)

        self.act_q4 = FastSign()
        self.conv4 = BinaryConv2d(in_channels = 384, out_channels = 384 ,kernel_size = (3, 3), padding = (1,1))
        # Now size is Nx384x7x7 
        self.bn4 = nn.BatchNorm2d(384, eps = eps, affine = affine, momentum = bn_mom)

        self.act_q5 = FastSign()
        self.conv5 = BinaryConv2d(in_channels = 384, out_channels = 256 ,kernel_size = (3, 3), padding = (1,1))
        # Now size is  Nx256x7x7
        self.bn5 = nn.BatchNorm2d(256, eps = eps, affine = affine, momentum = bn_mom)
       # After max_pool, size is Nx256x3x3

        # Stage 4
        self.act_fc1 = FastSign()
        self.fc1 = BinaryFully_connected(in_features = 5, out_features=4096 )
        # Now size is Nx256x3x4096
        self.bn6 = nn.BatchNorm2d(256, eps = eps, affine = affine, momentum = bn_mom)
        # Now size is Nx256x3x4096

        # Stage 5
        self.act_fc2 = FastSign()
        self.fc2 = BinaryFully_connected(in_features = 4096, out_features=4096 ) 
        # Now size is Nx256x3x4096
        self.bn7 = nn.BatchNorm2d(256, eps = eps, affine = affine, momentum = bn_mom)

        # Stage 6 
        self.fc3 = nn.Linear(in_features = 4096, out_features =num_classes )
        # Now size is Nx256x3xnum_classes 
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        # Stage 1 
        x=self.conv1(x)
        x=F.relu(self.bn1(x))
        # Now size is Nx96x62x62
        x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2))
        # Now size is Nx96x30x30

        # Stage 2 
        x=self.act_q2(x) 
        x=self.conv2(x) 
        x=self.bn2(x) 
        # Now size is Nx256x15x15
        x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2))
        # Now size is Nx256x7x7

        #Stage 3 
        x=self.act_q3(x) 
        x=self.conv3(x) 
        x=self.bn3(x) 
        x=self.act_q4(x) 
        x=self.conv4(x) 
        x=self.bn4(x) 
        x=self.act_q5(x) 
        x=self.conv5(x) 
        x=self.bn5(x) 
        # Now size is Nx256x7x7
        x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2))
        # Now size is Nx256x3x3

        #Stage 4 
        x=self.act_fc1(x) 
        x = self.fc1(x)
        x=F.relu(self.bn6(x))

        #Stage 5 
        x=self.act_fc2(x) 
        x = self.fc2(x) 
        x=F.relu(self.bn7(x))

        #Stage 6 
        x = self.fc3(x)
        x = self.softmax (x)
        return x 

#def simBNN1():
#    model = simBNN(num_classes =10)
#    return model

simBNN1 = simBNN(num_classes =10)

# Cuda the model 
#model = simBNN1()
# model.cuda() 

# Define a loss function and a optimizer
criterion = nn.CrossEntropyLoss()  #criterion = wage_util.SSE() 
optimizer = optim.SGD(simBNN1.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# Train the network 
for epoch in range(epoch_num):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = simBNN1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('loss:', '{', str(running_loss / 2000), '}')
            running_loss = 0.0

print('Finished Training')

# Save the tained model 
PATH = './simBNN_cifar10.pth'
torch.save(simBNN.state_dict(), PATH)

# Inference and Calculate accuracy 
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = simBNN1(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: ', str (100 * correct // total))