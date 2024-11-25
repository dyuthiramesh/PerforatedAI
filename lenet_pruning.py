import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchsummary import summary
from thop import profile
import os
import csv
import math
import time
from datetime import timedelta
import torch.nn.functional as F

# Initialize random seed for reproducibility
seed = 1787
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

# Set device
device = th.device("cuda" if th.cuda.is_available() else "cpu")

epochs = 100
prune_percentage = [0.04, 0.12]
prune_limits=[1,2]

trainloader = th.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', download=True, train=True,
                               transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=100, shuffle=True)

testloader = th.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', download=True, train=False,
                               transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=100, shuffle=True)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.a_type='relu'
        for m in self.modules():
            self.weight_init(m)
        self.softmax = nn.Softmax(dim=1)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=self.a_type)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layer1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        layer2 = F.max_pool2d(F.relu(self.conv2(layer1)), 2)
        layer2_p = layer2.view(-1, int(layer2.nelement() / layer2.shape[0]))
        layer3 = F.relu(self.fc1(layer2_p))
        layer4 = F.relu(self.fc2(layer3))
        layer5 = self.fc3(layer4)
        return layer5

class PruningMethod:
    def prune_filters(self, layer_indices):
        conv_layer = 0
        for layer_name, layer_module in self.named_modules():
            if isinstance(layer_module, th.nn.Conv2d):
                if conv_layer == 0:
                    in_channels = [i for i in range(layer_module.weight.shape[1])]
                else:
                    in_channels = layer_indices[conv_layer - 1]

                out_channels = layer_indices[conv_layer]
                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))

                if layer_module.bias is not None:
                    layer_module.bias = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))

                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.numpy()[:, in_channels])).to('cuda'))
                layer_module.in_channels = len(in_channels)
                layer_module.out_channels = len(out_channels)
                
                conv_layer += 1

            if isinstance(layer_module, th.nn.BatchNorm2d):
                out_channels = layer_indices[conv_layer]
                layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
                layer_module.bias = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))
                layer_module.running_mean = th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
                layer_module.running_var = th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')
                layer_module.num_features = len(out_channels)

            if isinstance(layer_module, nn.Linear):
                conv_layer -= 1
                in_channels = layer_indices[conv_layer]
                weight_linear = layer_module.weight.data.cpu().numpy()
                size = 4 * 4
                expanded_in_channels = []
                for i in in_channels:
                    for j in range(size):
                        expanded_in_channels.extend([i * size + j])
                layer_module.weight = th.nn.Parameter(th.from_numpy(weight_linear[:, expanded_in_channels]).to('cuda'))
                layer_module.in_features = len(expanded_in_channels)
                break

    def get_indices_topk(self, layer_bounds, i, prune_limit, prune_percentage):
        indices = int(len(layer_bounds) * prune_percentage[i]) + 1
        p = len(layer_bounds)
        if (p - indices) < prune_limit:
            remaining = p - prune_limit
            indices = remaining
        k = sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
        return k

    def get_indices_bottomk(self, layer_bounds, i, prune_limit):
        k = sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
        return k

class PruningLeNet(LeNet, PruningMethod):
    pass

# Load the model
model = PruningLeNet().to(device)

criterion = nn.CrossEntropyLoss()

checkpoint = th.load('lenet_base.pth')
model.load_state_dict(checkpoint['model'])

dummy_input = th.randn(1, 1, 28, 28).to(device)
initial_flops, initial_params = profile(model, inputs=(dummy_input,))

conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
#There are only 2 Conv layers, 20 filters and 50 filters

prunes = 0

# Create CSV file and write the header
with open('lenet.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'Epochs', 'CE Loss', 'Test Acc', 'Conv1', 'Conv2', 'Params', 'Params %','Flops', 'Flops %'])

continue_pruning = True
# The loop continues pruning until each layer has at least the number of filters defined by prune_limits
while continue_pruning:
    optimizer_pre_prune = th.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9,0.999), weight_decay=2e-4)
    scheduler_pre_prune = th.optim.lr_scheduler.MultiStepLR(optimizer_pre_prune, milestones=[20, 30], gamma=0.1)

    if prunes > 0:
        for epoch in range(epochs):
            train_acc = []
            old_running_loss, running_loss = 0, 0

            # Load model only if it exists
            checkpoint_path = f'test_model_prune_{prunes}.pth'
            if os.path.isfile(checkpoint_path):
                checkpoint = th.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
                optimizer_pre_prune.load_state_dict(checkpoint['optimizer'])
                scheduler_pre_prune.load_state_dict(checkpoint['scheduler'])
                best_train_acc = checkpoint['train_acc']
                best_test_acc = checkpoint['test_acc']

            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_pre_prune.zero_grad()

                output = model(inputs)
                old_loss = criterion(output, targets)
                old_running_loss += old_loss.item()

                new_loss = old_loss
                new_loss.backward()
                optimizer_pre_prune.step()
                running_loss += new_loss.item()

                with th.no_grad():
                    y_hat = th.argmax(output, 1)
                    train_acc.append((y_hat == targets).sum().item())

            epoch_train_acc = sum(train_acc) * 100 / len(trainloader.dataset)
            print(f'Epoch [{epoch+1}/{epochs}], Train Accuracy: {epoch_train_acc:.2f}%')
            print(f'Pruning Iteration {prunes + 1}, Epoch [{epoch + 1}/{epochs}], Old Loss: {old_running_loss / len(trainloader):.8f}, New Loss: {running_loss / len(trainloader):.8f}')

            test_acc = []
            with th.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    y_hat = th.argmax(output, 1)
                    test_acc.append((y_hat == targets).sum().item())
            epoch_test_acc = sum(test_acc) * 100 / len(testloader.dataset)
            print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {epoch_test_acc:.2f}%')

            if epoch == 0 or epoch_test_acc > best_test_acc:
                best_train_acc = epoch_train_acc
                best_test_acc = epoch_test_acc

                # Save each pruned model version separately
                th.save({'model': model.state_dict(),
                         'optimizer': optimizer_pre_prune.state_dict(),
                         'scheduler': scheduler_pre_prune.state_dict(),
                         'train_acc': best_train_acc,
                         'test_acc': best_test_acc,
                         'running_loss': running_loss,
                         'old_running_loss': old_running_loss}, f'test_model_prune_{prunes}.pth')

        # Reload best pruned model for accuracy tracking
        checkpoint = th.load(f'test_model_prune_{prunes}.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer_pre_prune.load_state_dict(checkpoint['optimizer'])
        scheduler_pre_prune.load_state_dict(checkpoint['scheduler'])
        best_train_acc = checkpoint['train_acc']
        best_test_acc = checkpoint['test_acc']
        best_running_loss = checkpoint['running_loss']
        best_old_running_loss = checkpoint['old_running_loss']

        csv_data = [prunes + 1, epochs, (best_old_running_loss / len(trainloader)), best_test_acc,
                    conv_layers[0].out_channels, conv_layers[1].out_channels]

    else:
        csv_data = [prunes + 1, epochs, '-', checkpoint['test_acc'], conv_layers[0].out_channels, conv_layers[1].out_channels]

    print(f'Conv 1 - Remaining Filters: {conv_layers[0].out_channels}')
    print(f'Conv 2 - Remaining Filters: {conv_layers[1].out_channels}')

    # Pruning filters with the lowest L2 norm
    selected_indices = [[] for _ in range(len(conv_layers))]
    remaining_indices = [[] for _ in range(len(conv_layers))]

    # Prune filters based on the percentage but never below the prune_limits
    for i, layer in enumerate(conv_layers):
        with th.no_grad():
            filters = layer.weight.data.clone()
            num_filters = filters.size(0)
    
            # Calculate L2 norms of each filter
            l2_norms = th.norm(filters.reshape(num_filters, -1), p=2, dim=1)
            sorted_indices = th.argsort(l2_norms)
    
            # Calculate the number of filters to prune, respecting the prune limits
            num_to_prune = max(1, int(prune_percentage[i] * num_filters))
            remaining_filters = num_filters - prune_limits[i]
            num_to_prune = min(num_to_prune, num_filters - prune_limits[i])  # Ensure we don't prune below the limit
    
            selected_indices[i] = sorted_indices[:num_to_prune].tolist()
            remaining_indices[i] = sorted_indices[num_to_prune:].tolist()

    print("Selected indices list:", selected_indices)
    print("Remaining indices list:", remaining_indices)

    flops, params = profile(model, inputs=(dummy_input,))
    print(f"Total FLOPs: {flops}, Total Params: {params}")
    csv_data += (params, ((initial_params - params) / initial_params) * 100, flops, ((initial_flops - flops) / initial_flops) * 100)

    with open('lenet.csv', mode='a', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(csv_data)

    print("\nPruning Starting")
    model.prune_filters(remaining_indices)  # This should implement pruning
    print("Pruning Done\n")
    if conv_layers[0].out_channels == prune_limits[0] and conv_layers[1].out_channels == prune_limits[1]:
        continue_pruning = False
    prunes += 1

print("Experiment completed successfully.")

# Print model summary
summary(model, (1, 28, 28))