import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.prune as prune
import torch
import time
#import pandas as pd
import numpy as np
import logging
import csv 
import math
from time import localtime, strftime
import os 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from itertools import zip_longest
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
import os
import random
from torch.utils.tensorboard import SummaryWriter

seed = 42
th.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

writer = SummaryWriter('resnet/pretrained')
device = 'cuda' if th.cuda.is_available() else 'cpu'

class Network():
    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented

class PruningMethod():
    
    def prune_filters(self,indices):
      conv_layer=0
      remaining_filters_count = []

      for layer_name, layer_module in self.named_modules():

        if(isinstance(layer_module, th.nn.Conv2d)  and layer_name!='conv1'):

          if(layer_name.find('conv1')!=-1):
            in_channels=[i for i in range(layer_module.weight.shape[1])]
            out_channels=indices[conv_layer]
            layer_module.weight = th.nn.Parameter( th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))

          if(layer_name.find('conv2')!=-1):
             in_channels=indices[conv_layer]
             out_channels=[i for i in range(layer_module.weight.shape[0])]
             layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[:,in_channels])).to('cuda'))
             conv_layer+=1

          # Track remaining filters in the current layer
          remaining_filters_count.append(len(out_channels))

          # Log remaining filters for the current layer
          writer.add_scalar(f'Remaining Filters Layer {conv_layer}', len(out_channels), epoch)
         
          layer_module.in_channels=len(in_channels)
          layer_module.out_channels=len(out_channels)
          

        if (isinstance(layer_module, th.nn.BatchNorm2d) and layer_name!='bn1' and layer_name.find('bn1')!=-1):
            out_channels=indices[conv_layer]


            layer_module.weight=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
            layer_module.bias=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
            
            layer_module.running_mean= th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
            layer_module.running_var=th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')
            
            layer_module.num_features= len(out_channels)

        writer.add_scalar('Total Remaining Filters', sum(remaining_filters_count), epoch)      
        if isinstance(layer_module, nn.Linear):

            break

 
    def get_indices_topk(self,layer_bounds,layer_num,prune_limit,prune_value):

      i=layer_num
      indices=prune_value[i]

      p=len(layer_bounds)
      if (p-indices)<prune_limit:
         prune_value[i]=p-prune_limit
         indices=prune_value[i]
      
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
      return k
      
    def get_indices_bottomk(self,layer_bounds,i,prune_limit):

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
      return k

norm_mean, norm_var = 0.0, 1.0


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBasicBlock(nn.Module,Network,PruningMethod):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes-inplanes-(planes//4)), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu2(out)

        return out


class ResNet(nn.Module,Network,PruningMethod):
    def __init__(self, block, num_layers, covcfg,num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        self.covcfg = covcfg
        self.num_layers = num_layers

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)


        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(1,block, 16, blocks=n, stride=1)
        self.layer2 = self._make_layer(2,block, 32, blocks=n, stride=2)
        self.layer3 = self._make_layer(3,block, 64, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if num_layers == 110:
            self.linear = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()
        self.layer_name_num={}
        self.pruned_filters={}
        self.remaining_filters={}

        self.remaining_filters_each_epoch=[]

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,a, block, planes, blocks, stride):
        layers = [] 

        layers.append(block(self.inplanes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layers == 110:
            x = self.linear(x)
        else:
            x = self.fc(x)

        return x


def resnet_56():
    cov_cfg = [(3 * i + 2) for i in range(9 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 56, cov_cfg)
def resnet_110(n_iterations):
    cov_cfg = [(3 * i + 2) for i in range(18 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 110, cov_cfg)
def resnet_32():
    cov_cfg = [16, 16, 32, 64]
    return ResNet(ResBasicBlock, 32, cov_cfg)

th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
N = 1

batch_size_tr = 64
batch_size_te = 64

epochs = 100
lr=0.01
milestones_array=[82,123]
lamda=0.000001
weight_decay=2e-4
momentum=0.9

# prune_limits=[6]*9*3
# prune_value=[1]*9+[2]*9+[4]*9

prune_limits=[6]*5*3
prune_value=[1]*5+[2]*5+[4]*5

total_layers=32
total_convs=15
total_blocks=3

# th.cuda.set_device(0)
gpu = th.cuda.is_available()

#model=resnet_56()

if not gpu:
    print('qqqq')
else:
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=batch_size_tr, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = th.utils.data.DataLoader(testset, batch_size=batch_size_te, shuffle=True, num_workers=2) 



# total_layers=56
# total_convs=9*3 #first conv also included
# total_blocks=3



decision_count=th.ones((total_convs))

short=False
tr_size = 50000
te_size=10000


activation = 'relu'

if gpu:
    model=resnet_56().cuda()
else:
    model=resnet_56()


criterion = nn.CrossEntropyLoss()

optimizer = th.optim.SGD(model.parameters(), lr=lr,momentum=0.9, weight_decay=2e-4,nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=milestones_array, gamma=0.1)

def evaluate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = total_loss / len(valid_loader)
    accuracy = 100 * correct / total
    return loss, accuracy


def custom_loss(outputs, labels, model, criterion, lambda_l1):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(param ** 2)
    # Cross-entropy loss
    ce_loss = criterion(outputs, labels)
    # Total loss with L1 regularization
    total_loss = ce_loss + (lambda_l1 * (math.exp(-math.log(l1_norm))))
    return total_loss

ans1='t'
if(ans1=='t'):
  checkpoint = th.load('resnet56_base.pth')
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  scheduler.load_state_dict(checkpoint['scheduler'])
  epoch_train_acc = checkpoint['train_acc']
  print(epoch_train_acc,'.......')
  epoch_test_acc = checkpoint['test_acc']
  print('model loaded')

elif(ans1=='f'):

    best_train_acc=0
    best_test_acc=0

    for n in range(1):

        mi_iteration=0
        for epoch in range(epochs):

          train_acc=[]
          for batch_num, (inputs, targets) in enumerate(trainloader):
            
            inputs = inputs.cuda()
            targets = targets.cuda()

            model.train()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            with th.no_grad():
              y_hat = th.argmax(output, 1)
              score = th.eq(y_hat, targets).sum()
              train_acc.append(score.item())

          
          with th.no_grad():
            epoch_train_acc=  (sum(train_acc)*100)/tr_size        
            test_acc=[]
            model.eval()
            for batch_nums, (inputs2, targets2) in enumerate(testloader):
                if(batch_nums==3 and short):
                    break

                inputs2, targets2 = inputs2.cuda(), targets2.cuda()            
                output=model(inputs2)
                y_hat = th.argmax(output, 1)
                score = th.eq(y_hat, targets2).sum()
                test_acc.append(score.item())

            epoch_test_acc= (sum(test_acc)*100)/te_size      
          writer.add_scalar('Training Loss', loss.item(), epoch)
          writer.add_scalar('Training Accuracy', epoch_train_acc, epoch)
          writer.add_scalar('Test Accuracy', epoch_test_acc, epoch)
          
          print('\n---------------Epoch number: {}'.format(epoch),
                  '---Train accuracy: {}'.format(epoch_train_acc),
                  '----Test accuracy: {}'.format(epoch_test_acc),'--------------')
          scheduler.step()
        #   print(optimizer.param_groups[0]['lr'])
else:
   print('wrong ans entered')
   import sys
   sys.exit()


a=[]
for layer_name, layer_module in model.named_modules():
  if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='conv1' and layer_name.find('conv1')!=-1):
    a.append(layer_module)

print("Training accuracy of baseline model is ",epoch_train_acc,'.......')
print("Testing accuracy of baseline model is ",epoch_test_acc,'.......')


# ended_epoch=0
best_train_acc=epoch_train_acc
best_test_acc=epoch_test_acc

epoch=0
writer.add_scalar('Baseline Accuracy/Train', epoch_train_acc, epoch)
writer.add_scalar('Baseline Accuracy/Test', epoch_test_acc, epoch)


with th.no_grad():

      #_______________________COMPUTE L1NORM____________________________________
    l1norm=[]
    l_num=0
    for layer_name, layer_module in model.named_modules():
          
        if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='conv1' and layer_name.find('conv1')!=-1):
            temp=[]
            filter_weight=layer_module.weight.clone()

            for k in range(filter_weight.size()[0]):
                temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))

            l1norm.append(temp)
            l_num+=1

    layer_bounds1=l1norm
#______Selecting__filters__to__regularize_____
    
    inc_indices=[]
    for i in range(len(layer_bounds1)):
        imp_indices=model.get_indices_bottomk(layer_bounds1[i],i,prune_limits[i])
        inc_indices.append(imp_indices)


    
    unimp_indices=[]
    dec_indices=[]
    for i in range(len(layer_bounds1)):
        temp=[]
        temp=model.get_indices_topk(layer_bounds1[i],i,prune_limits[i],prune_value)
        unimp_indices.append(temp[:])
        temp.extend(inc_indices[i])
        dec_indices.append(temp)
        
    # print('selected  UNIMP indices ',unimp_indices)

    remaining_indices=[]
    for i in range(total_convs):
      temp=[]
      for j in range(a[i].weight.shape[0]):
        if (j not in unimp_indices[i]):
          temp.extend([j])
      remaining_indices.append(temp)
continue_pruning=True
with th.no_grad():
      
    if(continue_pruning==True):
        model.prune_filters(remaining_indices)
        print(model)

ended_epoch=0
best_train_acc=0
best_test_acc=0


decision=True
best_test_acc= 0.0
prunes=1
lambda_l1 = 0.1

while(continue_pruning==True):

  if(continue_pruning==True):

     
    if(th.sum(decision_count)==0):
          continue_pruning=False 
    with th.no_grad():

      #_______________________COMPUTE L1NORM____________________________________
      l1norm=[]
      l_num=0
      for layer_name, layer_module in model.named_modules():
          
          if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='conv1' and layer_name.find('conv1')!=-1):
              temp=[]
              filter_weight=layer_module.weight.clone()

              for k in range(filter_weight.size()[0]):
                temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))

              l1norm.append(temp)
              l_num+=1

      layer_bounds1=l1norm

#______Selecting__filters__to__regularize_____
    
    inc_indices=[]
    for i in range(len(layer_bounds1)):
        imp_indices=model.get_indices_bottomk(layer_bounds1[i],i,prune_limits[i])
        inc_indices.append(imp_indices)
    
    unimp_indices=[]
    dec_indices=[]
    for i in range(len(layer_bounds1)):
        temp=[]
        temp=model.get_indices_topk(layer_bounds1[i],i,prune_limits[i],prune_value)
        unimp_indices.append(temp[:])
        temp.extend(inc_indices[i])
        dec_indices.append(temp)
        
    # print('selected  UNIMP indices ',unimp_indices)

    remaining_indices=[]
    for i in range(total_convs):
      temp=[]
      for j in range(a[i].weight.shape[0]):
        if (j not in unimp_indices[i]):
          temp.extend([j])
      remaining_indices.append(temp)

    if(continue_pruning==False):
       lamda=0
#______________________Custom_Regularize the model___________________________
    # if(continue_pruning==True):
    #    optimizer = th.optim.SGD(model.parameters(), lr=optim_lr,momentum=0.9)
    #    scheduler = MultiStepLR(optimizer, milestones=milestones_array, gamma=0.1)
    optimizer = th.optim.SGD(model.parameters(), lr=lr,momentum=0.9, weight_decay=2e-4,nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=milestones_array, gamma=0.1)

    epoch=0
    best_test_acc= 0.0
    for epoch in range(epochs):

        i=0
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if batch_idx==0:
                print("Loss cross-entropy = ",criterion(outputs,labels))
                l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                print("L1 NORM = ", l1_reg)
                print("New loss  = ",loss-(lamda*l1_reg))
            loss = custom_loss(outputs, labels, model, criterion, lambda_l1)
            l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            loss=loss-(lamda*l1_reg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = 100 * correct / total

        valid_loss, valid_acc = evaluate(model, testloader, criterion)
        end_time=time.time()
        total_time=end_time-start_time
        
        print('\nEpoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%, Time: {:.2f}s'.format(
            epoch + 1, epochs, epoch_loss, epoch_acc, valid_loss, valid_acc,total_time))
    

    writer.add_scalar(f'Pruning_{prunes}/Train Accuracy', epoch_acc, prunes)
    writer.add_scalar(f'Pruning_{prunes}/Test Accuracy', valid_acc, prunes)
    writer.add_scalar('Validation Loss: ', valid_loss, prunes)
    writer.add_scalar('Training Loss: ', epoch_loss, prunes)

    # Log the model structure (you can add it once or multiple times if the structure changes)
    writer.add_text(f'Pruning_{prunes}/Model Structure', str(model), prunes)

    with th.no_grad():
      #_______________________COMPUTE L1NORM____________________________________

      l1norm=[]
      l_num=0
      for layer_name, layer_module in model.named_modules():
                      
          if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='conv1' and layer_name.find('conv1')!=-1):
              temp=[]
              filter_weight=layer_module.weight.clone()              
              for k in range(filter_weight.size()[0]):
                      temp.append(float("{:.6f}".format((filter_weight[k,:,:,:]).norm(1).item())))
              l1norm.append(temp)
              l_num+=1

      layer_bounds1=l1norm  

    with th.no_grad():
      
      if(continue_pruning==True):
        model.prune_filters(remaining_indices)
        print(model)
      else:
        break

      #_________________________PRUNING_EACH_CONV_LAYER__________________________
      for i in range(len(layer_bounds1)):
          if(a[i].weight.shape[0]<= prune_limits[i]):
            decision_count[:]=0
            break
 
      prunes+=1
