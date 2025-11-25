# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        return x

# Training Neural Networks

# Import statements
import torch.optim as optim
import torch.nn.functional as F


# Create an instance of the network
net = SimpleCNN()


# Define a loss function
criterion = nn.CrossEntropyLoss()


# Define an optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Dummy dataset
inputs = torch.randn(100, 3, 32, 32)
labels = torch.randint(0, 10, (100,))


# Training loop
for epoch in range(10):
    running_loss = 0.0
    for i in range(100):
        inputs, labels = inputs, labels

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        if i % 10 == 9:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

