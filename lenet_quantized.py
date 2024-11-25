import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader
import os

# Define the LeNet5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(50 * 4 * 4, 800)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(800, 500)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = self.relu2(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 50 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss/len(train_loader), 100.*correct/total

# Function to evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Function to print file size in MB
def print_file_size(file_path):
    size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"File size of {file_path}: {size:.2f} MB")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)

    # Training loop
    best_accuracy = 0.0
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, epoch)
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        
        # Print epoch-wise test accuracy
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
        
        # Update learning rate
        scheduler.step(test_accuracy)
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'lenet5_best.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load('lenet5_best.pth'))
    
    # Save the final original model
    torch.save(model.state_dict(), 'lenet5_original.pth')
    
    # Final evaluation of original model
    print("\nEvaluating final original model...")
    original_accuracy = evaluate_model(model, test_loader, device)

    # Perform post-training dynamic quantization
    print("\nQuantizing model...")
    quantized_model = quantize_dynamic(
        model.cpu(),  # Quantization requires CPU
        {nn.Linear, nn.Conv2d},  # Specify which layers to quantize
        dtype=torch.qint8
    )

    # Save the quantized model
    torch.save(quantized_model.state_dict(), 'lenet5_quantized.pth')

    # Evaluate quantized model
    print("\nEvaluating quantized model...")
    quantized_accuracy = evaluate_model(quantized_model, test_loader, torch.device('cpu'))

    # Print final results
    print("\nFinal Results:")
    print(f"Best accuracy achieved during training: {best_accuracy:.2f}%")
    print(f"Original model final accuracy: {original_accuracy:.2f}%")
    print(f"Quantized model accuracy: {quantized_accuracy:.2f}%")
    
    print("\nModel file size comparison:")
    print_file_size('lenet5_original.pth')
    print_file_size('lenet5_quantized.pth')

if __name__ == '__main__':
    main()
