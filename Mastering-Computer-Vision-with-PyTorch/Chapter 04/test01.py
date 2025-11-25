# -*- coding: utf-8 -*-
# Processing of Information by Neural Networks

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(3, 5)  # Input layer with 3 neurons and hidden layer with 5 neurons
        self.layer2 = nn.Linear(5, 1)  # Hidden layer with 5 neurons and output layer with 1 neuron
        self.activation = nn.ReLU()    # Activation function


    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


# Create an instance of the SimpleNN class
model = SimpleNN()


# Generate random input data
input_data = torch.randn(1, 3)


# Forward pass through the network
output = model(input_data)
print(output)


