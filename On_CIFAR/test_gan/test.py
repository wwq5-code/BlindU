import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define a function to store the gradients
def store_gradients(grad, name):
    gradients[name] = grad

# Register hooks for each layer you're interested in
def register_hooks(module):
    for name, layer in module.named_modules():
        layer.register_backward_hook(lambda grad, _, __: store_gradients(grad, name))

# Initialize ResNet18 model
resnet18 = models.resnet18(pretrained=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18.to(device)

# Register the hooks for each layer
gradients = {}
register_hooks(resnet18)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        # Send inputs and targets to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = resnet18(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Print the gradients of each layer
        print(f"Batch {i + 1}: Gradients of each layer:")
        for name, grad in gradients.items():
            print(f"{name}: {grad}")

        # Reset gradients storage
        gradients = {}
