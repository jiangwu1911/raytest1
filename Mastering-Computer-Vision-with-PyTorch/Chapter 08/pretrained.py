import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn

# Data preparation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Load the pre-trained model and modify it for CIFAR-10
model = resnet50(weights="IMAGENET1K_V2")

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze some layers
for param in list(model.layer4.parameters()) + list(model.fc.parameters()):
    param.requires_grad = True

# Modify the last layer
model.fc = nn.Linear(model.fc.in_features, 10)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# Save the trained model
torch.save(model.state_dict(), 'finetuned_resnet50_cifar10.pth')

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # No need to track the gradients
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

# Calculate accuracy on the validation set
val_accuracy = calculate_accuracy(testloader, model)
print(f'Accuracy of the netowrk on the test images: {val_accuracy:.2f}')



