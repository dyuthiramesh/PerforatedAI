import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import math
import numpy as np
from thop import profile
from torch.utils.tensorboard import SummaryWriter
# from important_filters import Network, get_important_filters
import random

random_seed=42
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
pruning=0
last_layer=0
writer = SummaryWriter("run/complete2")

import os
import csv

class CIFAR10VGG(nn.Module):
    def __init__(self):
        super(CIFAR10VGG, self).__init__()
        self.num_classes = 10
        self.weight_decay = 0.0005

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.4)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(0.4)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.dropout5 = nn.Dropout(0.4)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.dropout6 = nn.Dropout(0.4)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.dropout7 = nn.Dropout(0.4)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        self.dropout8 = nn.Dropout(0.4)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout9 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool3(x)

        x = F.relu(self.bn8(self.conv8(x)))
        x = self.dropout5(x)
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.dropout6(x)
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.pool4(x)

        x = F.relu(self.bn11(self.conv11(x)))
        x = self.dropout7(x)
        x = F.relu(self.bn12(self.conv12(x)))
        x = self.dropout8(x)
        x = F.relu(self.bn13(self.conv13(x)))
        x = self.pool5(x)
        x = self.dropout9(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x

def calculate_top_filters(base_model, train_loader, conv_layers):
    batch_size = 128 
    network = Network(base_model)

    layer_votes_batch = {}
    layer_aggregated_votes = {}

    for conv in conv_layers:
        layer_votes_batch[conv] = []

    for batch_images, batch_labels in train_loader:

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        layer_votes = network.cal_mi(batch_images, batch_labels, conv_layers)

        for layer in conv_layers:
            layer_votes_batch[layer].append(layer_votes[layer])


    for layer in conv_layers:
        layer_aggregated_votes[layer] = np.mean(layer_votes_batch[layer], axis=0)

    return layer_aggregated_votes

def sort(layer_aggregated_votes, conv_layers, k):
    layer_result = {}
    for layer in conv_layers:
        agg_votes = layer_aggregated_votes[layer]
        k_indices=int((agg_votes.size) * k)
        flat_indices = np.argsort(agg_votes.ravel())[-k_indices:]
        indices = np.unravel_index(flat_indices, agg_votes.shape)
        result = np.zeros_like(agg_votes)
        result[indices] = 1
        layer_result[layer] = result

    return layer_result

def get_important_filters(model, train_loader):
    
    conv_layers = []
    for name,layer in model.named_children():
        if "conv" in name:
            conv_layers.append(str(name))

    most_important_filters = calculate_top_filters(model, train_loader, conv_layers)
    
    sorted_filters = sort(most_important_filters, conv_layers, 0.25)
    return sorted_filters, conv_layers

params={
  "train_batch_size":128,
  "test_batch_size":128,
  "learning_rate":0.1,
  "num_epochs":250,
  "pruning_rate":0.02,
  "lambda_l1":10000,
}

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data normalization (mean and std for CIFAR-10)
def normalize_cifar10(train_data, test_data):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform_train, transform_test

def custom_loss(outputs, labels, model, criterion, lambda_l1):
    '''
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    '''
    l2 = 0
    for param in model.parameters():
        l2 += torch.sum(param ** 2)
    ce_loss = criterion(outputs, labels)
     
    total_loss = ce_loss + (lambda_l1 * (math.exp(-math.log(l2))))
    #total_loss = ce_loss + (lambda_l1 * (math.exp(-math.log(l1_norm))))
    # print("ce loss ",ce_loss)
    # print("Regularizer ",(lambda_l1 * (math.exp(-math.log(l1_norm)))))
    # print("Total Loss ",total_loss)
    return total_loss 

def train_model(model, train_loader, test_loader, epochs=250, lr=0.1, lr_drop=20, base=False):
    global pruning
    best_accuracy=0
    best_model=None
    pruning += 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop, gamma=0.5)
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            '''
            if base:
                loss = criterion(outputs, targets)
            if not base:
                loss=custom_loss(outputs, targets, model, criterion, params["lambda_l1"])
            '''
            loss.backward()
            
            '''
            for layer_name in conv_layers:
                conv_layer = getattr(model, layer_name)
                filters_state = list(selected_filters[layer_name])
                filter_mask = torch.tensor(filters_state, dtype=torch.float32).view(-1, 1, 1, 1).to(device)

                with torch.no_grad():
                    conv_layer.weight.grad *= filter_mask
            '''
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)  # Multiply by batch size to sum all loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100. * correct / total
        avg_train_loss = train_loss / total

        test_loss, test_accuracy = test_model(model, test_loader, criterion)  # Test at the end of each epoch
        input = torch.randn(1, 3, 32, 32).to(device)
        flops, params = profile(model, inputs=(input, ))

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        writer.add_scalar(f'Loss/Train {pruning}Prune', train_loss, epoch)
        writer.add_scalar(f'Accuracy/Train {pruning}Prune', train_accuracy, epoch)
        writer.add_scalar(f'Loss/Test {pruning}Prune', test_loss, epoch)
        writer.add_scalar(f'Accuracy/Test {pruning}Prune', test_accuracy, epoch)
        
        scheduler.step()
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = deepcopy(model)

    return best_model

def test_model(model, test_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)  # Multiply by batch size to sum all loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_test_loss = test_loss / total
    test_accuracy = 100. * correct / total
    return avg_test_loss, test_accuracy


def calculate_l1_norm_of_linear_outputs(model):
    l1_normalisation_values = {}
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            l1_norm_of_neurons = torch.sum(torch.abs(weights), dim=1).tolist()
            l1_normalisation_values[name] = l1_norm_of_neurons
    return l1_normalisation_values

def calculate_l1_norm_of_linear_inputs(model):
    l1_normalisation_values = {}
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            l1_norm_of_inputs = torch.sum(torch.abs(weights), dim=0).tolist()
            l1_normalisation_values[name] = l1_norm_of_inputs
    return l1_normalisation_values


def calculate_threshold_l1_norm(values, percentage_to_prune):
    threshold_values = {}
    for layer_name, vals in values.items():
        sorted_vals = sorted(vals)
        threshold_index = int(len(sorted_vals) * percentage_to_prune)
        threshold_value = sorted_vals[threshold_index]
        threshold_values[layer_name] = threshold_value
    return threshold_values

def print_conv_layer_shapes(model):
    print("\nLayer and shape of the filters \n -----------------------------")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Conv layer: {name}, Weight shape: {module.weight.shape}  Bias shape: {module.bias.shape if module.bias is not None else 'No bias'}")

def calculate_l1_norm_of_filters(model):
    l1_normalisation_values={}
    for name,layer in model.named_children():
        if isinstance(layer,nn.Conv2d):
            filters=layer.weight
            l1_norm_of_filter=[]
            for idx,filter in enumerate(filters):
                l1_norm=torch.sum(torch.abs(filter)).item()
                l1_norm_of_filter.append(l1_norm)
            l1_normalisation_values[name]=l1_norm_of_filter
    return l1_normalisation_values

def calculate_threshold_l1_norm_of_filters(l1_normalisation_values,percentage_to_prune):
    threshold_values={}
    for filter_ in l1_normalisation_values:
        filter_values=l1_normalisation_values[filter_]
        sorted_filter_values=sorted(filter_values)
        threshold_index=int(len(filter_values)*percentage_to_prune)
        threshold_value=sorted_filter_values[threshold_index]
        threshold_values[filter_]=threshold_value
    return threshold_values

def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor


def get_new_conv(in_channels, conv, dim, channel_index, independent_prune_flag=False):
  
    new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

    new_conv.weight.data = index_remove(conv.weight.data, 0, channel_index)
    # new_conv.bias.data = index_remove(conv.bias.data, 0, channel_index)

    return new_conv

def prune_layer(layer, outputs_to_prune, inputs_to_prune):
    in_features = layer.in_features - len(inputs_to_prune)
    out_features = layer.out_features - len(outputs_to_prune)

    new_linear_layer = nn.Linear(in_features, out_features, bias=True)

    keep_outputs = list(set(range(layer.out_features)) - set(outputs_to_prune))
    keep_inputs = list(set(range(layer.in_features)) - set(inputs_to_prune))


    new_linear_layer.weight.data = layer.weight.data[keep_outputs][:, keep_inputs]
    new_linear_layer.bias.data = layer.bias.data[keep_outputs]
    
    output_weights=new_linear_layer.out_features
    return new_linear_layer,output_weights

def prune_filters(model,threshold_values,l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs):
    global last_layer
    filters_to_remove=[]
    next_channel=3
    for name,layer in model.named_children():
        filters_to_remove=[]
        if isinstance(layer,nn.Conv2d):
            filters=layer.weight
            num_filters_to_prune=0
            print(threshold_values[name])
            for idx, filter in enumerate(filters):
                l1_norm = torch.sum(torch.abs(filter)).item()
                if l1_norm < threshold_values[name]:
                    num_filters_to_prune+=1
                    layer.weight.data[idx].zero_()
                    filters_to_remove.append(idx)
            
            if num_filters_to_prune == 0:
                for idx, filter in enumerate(filters):
                    l1_norm = torch.sum(torch.abs(filter)).item()
                    if l1_norm <= threshold_values[name]:
                        num_filters_to_prune+=1
                        layer.weight.data[idx].zero_()
                        filters_to_remove.append(idx)

            if num_filters_to_prune > 0:
                in_channels = next_channel
                out_channels = layer.out_channels - num_filters_to_prune
                new_conv_layer=get_new_conv(in_channels,layer,0,filters_to_remove).to(device)
                setattr(model, name, new_conv_layer)
                next_channel=out_channels

        elif isinstance(layer, nn.BatchNorm2d):
            new_batch_norm_2d_layer=nn.BatchNorm2d(num_features=next_channel).to(device)
            setattr(model,name,new_batch_norm_2d_layer)
            del new_batch_norm_2d_layer

        elif isinstance(layer, nn.BatchNorm1d):
            new_batch_norm_1d_layer=nn.BatchNorm1d(num_features=next_channel).to(device)
            setattr(model,name,new_batch_norm_1d_layer)
            del new_batch_norm_1d_layer

        elif isinstance(layer, nn.Linear):
            if layer==last_layer:
                outputs_to_prune=[]
            else:
                outputs_to_prune = [idx for idx, l1 in enumerate(l1_norm_outputs[name]) if l1 < threshold_outputs[name]]
            inputs_to_prune = [idx for idx, l1 in enumerate(l1_norm_inputs[name]) if l1 < threshold_inputs[name]]
            new_layer,next_channel= prune_layer(layer, outputs_to_prune, inputs_to_prune)
            setattr(model, name, new_layer)
    return model

def update_inputs_channels(model):
    prev_channels=3
    for name,module in model.named_children():
        if isinstance(module,nn.Conv2d):
            in_channels=prev_channels
            module.weight.data = module.weight.data[:, :in_channels, :, :]
            module.in_channels=in_channels
            prev_channels=module.out_channels
    return model

def prune_model(model,pruning_rate,l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs):
    l1_norm_values=calculate_l1_norm_of_filters(model)
    threshold_values=calculate_threshold_l1_norm_of_filters(l1_norm_values,pruning_rate)
    model=prune_filters(model,threshold_values,l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs)
    model=update_inputs_channels(model)
    return model

def check_pruning(model):
  print("\nLayer and filter sizes \n ------------------------------------")
  for name,module in model.named_modules():
    if isinstance(module,nn.Conv2d):
      print(f"Layer: {name}, Filter Size: {module.out_channels}")


def l1_norm(model):
    '''
    l1 = 0
    for param in model.parameters():
        l1 += torch.sum(torch.abs(param))
    return l1
    '''
    l2 = 0
    for param in model.parameters():
        l2 += torch.sum(param ** 2)
    return torch.sqrt(l2)

def complete_train(model):
  
    l1_norm_outputs = calculate_l1_norm_of_linear_outputs(model)
    l1_norm_inputs = calculate_l1_norm_of_linear_inputs(model)
    threshold_outputs = calculate_threshold_l1_norm(l1_norm_outputs, params["pruning_rate"])
    threshold_inputs = calculate_threshold_l1_norm(l1_norm_inputs, params["pruning_rate"])

    print("\nBefore pruning:\n")
    print_conv_layer_shapes(model)

    model=prune_model(model,params["pruning_rate"],l1_norm_inputs,l1_norm_outputs,threshold_inputs,threshold_outputs)

    print("\nAfter pruning:\n")
    print_conv_layer_shapes(model)

    print("\n Pruned Filter Sizes \n")
    check_pruning(model)
    
    print("The model that we are using is \n",model)
    l1_pre_maximising=l1_norm(model)
    print(f"\n\n Pre training L1 Norm: {l1_pre_maximising}\n\n")

    # model=train(model, criterion, optimizer, scheduler, train_loader, test_loader, params["num_epochs"], params["lambda_l1"] )
    # train_model(model, train_loader, test_loader, epochs=250, lr=0.1, lr_drop=20, base=False, selected_filters=None, conv_layers = None)
    
    #filter_importance, conv_layers = get_important_filters(model, train_loader)
    
    #model=train_model(model = model, train_loader=train_loader, test_loader=test_loader, selected_filters=filter_importance, conv_layers=conv_layers)
    model=train_model(model = model, train_loader=train_loader, test_loader=test_loader, epochs = 250)
    l1_post_maximising=l1_norm(model)
    print(f"\n\nPost training L1 Norm: {l1_post_maximising}\n\n")
    return model

def prune(model):
    global pruning, last_layer
    csv_file = "vgg16_pruning_pretrained.csv"
    
    # Write header to the CSV file if it doesn't exist or is empty
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6", "Layer 7", "Layer 8", "Layer 9", "Layer 10", "Layer 11", "Layer 12", "Layer 13",
                         "Test Accuracy", "FLOPs", "Parameters"])
    
    for iteration in range(65):
        input = torch.randn(1, 3, 32, 32).to(device)
        macs, params = profile(model, inputs=(input,))
        print("Macs ", macs)
        print("Params ", params)

        # Extract number of filters in each layer
        num_filters = []
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                num_filters.append(layer.out_channels)
        
        # Run training/testing and compute test accuracy
        model = complete_train(model)
        test_loss, test_accuracy = test_model(model, test_loader, nn.CrossEntropyLoss())

        # Save model checkpoint
        torch.save(model, f"complete3/{pruning}_pruned_model.pth")

        # Append data to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([pruning] + num_filters + [test_accuracy, macs, params])
        pruning += 1
    
    return model

if __name__ == '__main__':
    transform_train, transform_test = normalize_cifar10(None, None)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = CIFAR10VGG().to(device)
    model.load_state_dict(torch.load("vgg16_pretrained.pth"))
    torch.save(model, f'complete2/start_model.pth')
    l1_norm_of_initial_model=l1_norm(model)
    print("L1 norm of model initially is ",l1_norm_of_initial_model)
    
    layers = list(model.modules())
    last_layer=layers[-1]
    # Train and save the model
    # base_model=train_model(model, train_loader, test_loader)
    model=prune(model)
    torch.save(model, f'complete2/pruned_model.pth')
    
    l1_norm_of_pruned_model=l1_norm(model)
    print("L1 norm of model pruned is ",l1_norm_of_pruned_model)
    # Load the model and evaluate
    # model.load_state_dict(torch.load('cifar10vgg.pth'))
    # test_model(model, test_loader)
    writer.close()
