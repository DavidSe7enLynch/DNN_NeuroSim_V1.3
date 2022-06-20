import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch
from models import dataset
import torch.optim as optim
from torch.autograd import Variable
import time 

# Assume the input is [4, 3, 32, 32]

# Global Variables 
in_channels = 3 
batch_size = 4
lr=0.001
epoch_num = 5 #7
seed = 117 

eps = 0.00002 
bn_mom = 0.9
affine = True 

# Load Data Set [4, 3, 32, 32]
#train_loader, test_loader = dataset.get_imagenet(batch_size=batch_size, num_workers=1)
train_loader, test_loader = dataset.get_cifar10(batch_size=batch_size, num_workers=2)

classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

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
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            FastSign(),
            BinaryConv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            FastSign(),
            BinaryConv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            FastSign(),
            BinaryConv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            FastSign(),
            BinaryConv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            FastSign(),
            BinaryFully_connected(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            FastSign(),
            BinaryFully_connected(4096, 4096),
            nn.ReLU(inplace=True),
            FastSign(),
            BinaryFully_connected(4096, num_classes),
        )

        # Stage 1 
        #self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64 ,kernel_size = (11, 11),stride = (4, 4), padding = (2,2))
        # Now size is [[4, 64, 32, 32]
        #self.bn1 = nn.BatchNorm2d(96, eps = eps, affine = affine, momentum = bn_mom)
        # At the end of stage 1, size is 

        # Stage 2 
        #self.act_q2 = FastSign()
        #self.conv2 = BinaryConv2d(in_channels = 96, out_channels = 256 ,kernel_size = (5, 5), padding = (2,2)) 
        # Now size is 
        #self.bn2 = nn.BatchNorm2d(256, eps = eps, affine = affine, momentum = bn_mom)
        # At the end of stage 2, size is 

        # Stage 3 
        #self.act_q3 = FastSign()
        #self.conv3 = BinaryConv2d(in_channels = 256, out_channels = 384 ,kernel_size = (3, 3), padding = (1,1))
        # Now size is 
        #self.bn3 = nn.BatchNorm2d(384, eps = eps, affine = affine, momentum = bn_mom)

        #self.act_q4 = FastSign()
        #self.conv4 = BinaryConv2d(in_channels = 384, out_channels = 384 ,kernel_size = (3, 3), padding = (1,1))
        # Now size is 
        #self.bn4 = nn.BatchNorm2d(384, eps = eps, affine = affine, momentum = bn_mom)

        #self.act_q5 = FastSign()
        #self.conv5 = BinaryConv2d(in_channels = 384, out_channels = 256 ,kernel_size = (3, 3), padding = (1,1))
        # Now size is 
        #self.bn5 = nn.BatchNorm2d(256, eps = eps, affine = affine, momentum = bn_mom)
        # At the end of stage 3, size is 

        # Stage 4
        #self.flatten = nn.Flatten(1,3)
        # Now size is 
        #self.act_fc1 = FastSign()
        #self.fc1 = BinaryFully_connected(in_features = 6400, out_features=4096 )
        # Now size is 
        #self.bn6 = nn.BatchNorm1d(4096, eps = eps, affine = affine, momentum = bn_mom) 
        # At the end of stage 4, size is 

        # Stage 5
        #self.act_fc2 = FastSign()
        #self.fc2 = BinaryFully_connected(in_features = 4096, out_features=4096 ) 
        # Now size is 
        #self.bn7 = nn.BatchNorm1d(4096, eps = eps, affine = affine, momentum = bn_mom)
        # At the end of stage 5, size is

        # Stage 6 
        #self.fc3 = nn.Linear(in_features = 4096, out_features =num_classes )
        # Now size is 
        #self.softmax = nn.Softmax(dim=1)
        # At the end of stage 6, size is

    def forward(self,x):
        #print ("original x is : ", x.size())
        # Stage 1 
        #x=self.conv1(x)
        #print ("after conv1, x is : ", x.size())
        #x=F.relu(self.bn1(x))
        #x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2), ceil_mode=False)
        #print ("At the end of stage 1,  x is : ", x.size())
        # Now size is [64, 96, 26, 26]

        # Stage 2 
        #x=self.act_q2(x) 
        #x=self.conv2(x) 
        #print ("After Conv2,  x is : ", x.size())
        #x=self.bn2(x) 
        #print ("Before maxpool,  x is : ", x.size())
        # Now size is [64, 256, 26, 26]
        #x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2))
        #print ("At the end of stage 2,  x is : ", x.size())
        # Now size is [64, 256, 12, 12]

        #Stage 3 
        #x=self.act_q3(x) 
        #x=self.conv3(x) 
        #print ("After Conv3,  x is : ", x.size())
        #x=self.bn3(x) 
        #x=self.act_q4(x) 
        #x=self.conv4(x) 
        #print ("After Conv4,  x is : ", x.size())
        #x=self.bn4(x) 
        #x=self.act_q5(x) 
        #x=self.conv5(x) 
        #print ("After Conv5,  x is : ", x.size())
        #x=self.bn5(x) 
        #print ("before maxpool,  x is : ", x.size())
        # Now size is [64, 256, 12, 12]
        #x=F.max_pool2d(x,kernel_size = (3, 3),stride = (2, 2))
        #print ("At the end of stage 3,  x is : ", x.size())
        # Now size is [64, 256, 5, 5]

        #Stage 4 
        #x = self.flatten(x) 
        #print ("after flatten,  x is : ", x.size())
        #x = self.act_fc1(x) 
        #x = self.fc1(x)
        #print ("after fc1,  x is : ", x.size())
        #x=F.relu(self.bn6(x))
        #print ("At the end of stage 4,  x is : ", x.size())

        #Stage 5 
        #x=self.act_fc2(x) 
        #x = self.fc2(x) 
        #print ("after fc2,  x is : ", x.size())
        #x=F.relu(self.bn7(x))
       # print ("At the end of stage 5,  x is : ", x.size())

        #Stage 6 
        #x = self.fc3(x)
        #print ("after fc3,  x is : ", x.size())
        #x = self.softmax (x)
        #print ("At the end of stage 6,  x is : ", x.size())
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x 


simBNN1 = simBNN(num_classes =10)

# cuda the model 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
simBNN1.to(device)

# Define a loss function and a optimizer
criterion = nn.CrossEntropyLoss()
# criterion = wage_util.SSE() 
optimizer = optim.SGD(simBNN1.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# Train the network 
for epoch in range(epoch_num):  # loop over the dataset multiple times

    running_loss = 0.0
    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
    #for i, (inputs, labels) in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = simBNN1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #Time
        end_time = time.time()
        time_taken = end_time - start_time

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            print('Time:',time_taken)
            running_loss = 0.0

print('Finished Training')

# Save the tained model 
PATH = './simBNN_cifar10.pth'
torch.save(simBNN1.state_dict(), PATH)

simBNN1 = simBNN(num_classes =10)
simBNN1.load_state_dict(torch.load(PATH))

# Inference and Calculate accuracy 
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i, data in enumerate(test_loader,0):
        images, labels = data[0].to(device), data[1].to(device)
        #images, labels = data
        # calculate outputs by running images through the network
        outputs = simBNN1(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: ', str (100 * correct // total))