import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch
from models import dataset
import torch.optim as optim
import time 
import torchvision.transforms as transforms
import torchvision

# Assume the input is [4, 3, 32, 32]

# Global Variables 
in_channels = 3 
batch_size = 4
lr=0.001
epoch_num = 5 
seed = 117 

# Load Data Set [4, 3, 32, 32]
#train_loader, test_loader = dataset.get_cifar10(batch_size=batch_size, num_workers=2)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

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
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #FastSign(),
            BinaryConv2d(64, 192, kernel_size=5, padding=2),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #FastSign(),
            BinaryConv2d(192, 384, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #FastSign(),
            BinaryConv2d(384, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #FastSign(),
            BinaryConv2d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            #FastSign(),
            BinaryFully_connected(256 * 6 * 6, 4096, bias=True),
            #nn.ReLU(inplace=True),
            nn.Dropout(),
            #FastSign(),
            BinaryFully_connected(4096, 1024, bias=True),
            #nn.ReLU(inplace=True),
            #FastSign(),
            BinaryFully_connected(1024, num_classes, bias=True),
        )

    def forward(self,x):
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
#PATH = './simBNN_cifar10.pth'
#torch.save(simBNN1.state_dict(), PATH)

#simBNN1 = simBNN(num_classes =10)
#simBNN1.load_state_dict(torch.load(PATH))

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

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for i, data in enumerate(test_loader,0):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = simBNN1(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))