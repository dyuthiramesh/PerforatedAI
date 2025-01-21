import argparse
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
from thop import profile
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from random import SystemRandom
import quantization as q

if __name__ == '__main__':
    # experiment_id = int(SystemRandom().random()*100000)
    experiment_id = 1
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    
    data_root = "./data"
    emnist_split = "balanced"
    batch_size = 100

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # DataLoaders
    trainloader = th.utils.data.DataLoader(
        datasets.EMNIST(
            root=data_root, split=emnist_split, train=True, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True
    )

    testloader = th.utils.data.DataLoader(
        datasets.EMNIST(
            root=data_root, split=emnist_split, train=False, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=False
    )


    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(50 * 4 * 4, 800)
            self.fc2 = nn.Linear(800, 500)
            self.fc3 = nn.Linear(500, 47)  # 47 classes for EMNIST Balanced
            self.a_type = 'relu'
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

    def evaluate(model, dataloader, device):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with th.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy
    
    model = LeNet().to(device)
    
    # checkpoint = th.load('lenet_emnist_best.pth')
    # model.load_state_dict(checkpoint['model'])
    # torch.save(model,"lenet_model_emnist_original_trained.h5")
    # torch.save(model,"lenet_model_emnist.pth")

    model = torch.load("lenet_model_emnist.pth")

    
    model.eval()
    
    
    original_accuracy = evaluate(model, testloader, device)

    print(f"Original Model Accuracy: {original_accuracy:.2f}%")

    model_weights, model_scales, model_zero_points = q.quantize_model(model)
    # print(model_weights)
    torch.save({
        "model_state_dict": model_weights,
        "model_scales": model_scales,
        "model_zero_points": model_zero_points
        },f"quantized_model_lenet.pth")
    
    model = LeNet().to(device)
    
    quantized_model = q.load_quantized_model(model)
    quantized_model.eval()
    quantized_accuracy = evaluate(quantized_model, testloader, device)

    print(f"Quantized Model Accuracy: {quantized_accuracy:.2f}%")

    # torch.save(quantized_model, "emnist_lenet_quantized_custom_model.pth")