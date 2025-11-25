# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms


# Set up the transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Define the neural network
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = SimpleCNN()


# Compile the model with torch.compile for PyTorch 2.0
compiled_net = torch.compile(net)


# Set up the loss function and optimizer
import torch.optim as optim


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(compiled_net.parameters(), lr=0.001, momentum=0.9)


# Train the network
for epoch in range(5):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = compiled_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


print('Finished Training')


# Test the network on the test data
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = compiled_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Evaluating Model performance

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Creating Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

y_pred = []
y_true = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Calculate Precision, Recall & F1 Score

from sklearn.metrics import precision_recall_fscore_support

# Prepare for metric calculation
all_labels = np.array([])
all_preds = np.array([])
all_outputs = np.array([])


# Test the network on the test data
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = compiled_net(images)
        _, predicted = torch.max(outputs.data, 1)

        # Store predictions, labels and output scores for metrics
        all_labels = np.append(all_labels, labels.numpy())
        all_preds = np.append(all_preds, predicted.numpy())
        all_outputs = np.append(all_outputs, outputs.numpy())


# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
print(f' Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

# Create ROC-AUC Curve

from sklearn.metrics import roc_auc_score, roc_curve, auc

# Prepare for metric calculation
all_labels = np.array([])
all_outputs = np.array([])
# Test the network on the test data
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = compiled_net(images)
        _, predicted = torch.max(outputs.data, 1)

        # Store predictions, labels and output scores for metrics
        all_labels = np.append(all_labels, labels.numpy())
        all_outputs = np.append(all_outputs, outputs.numpy())


# One-hot encode labels for ROC-AUC
one_hot_labels = np.eye(10)[all_labels.astype(int)]
roc_auc = roc_auc_score(one_hot_labels, all_outputs.reshape(-1, 10), multi_class='ovr')
print(f'ROC-AUC: {roc_auc:.2f}')
# Plot ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i], all_outputs.reshape(-1, 10)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()
for i in range(10):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC curve per class')
plt.legend(loc="lower right")
plt.show()

