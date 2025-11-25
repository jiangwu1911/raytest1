import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import ray

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

@ray.remote(num_gpus=0.1)
class RemoteTrainer:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = Net().to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)
        
    def train_epoch(self, epoch, train_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        self.scheduler.step()
        # Move model state to CPU before returning
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def get_model_state(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def set_model_state(self, state_dict):
        # Move state dict to GPU before loading
        state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def average_model_states(states_list):
    averaged_state = {}
    for key in states_list[0].keys():
        stacked_params = torch.stack([state[key] for state in states_list])
        averaged_state[key] = torch.mean(stacked_params, dim=0)
    return averaged_state

def main():
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    lr = 1.0
    gamma = 0.7
    no_cuda = False
    seed = 1
    num_trainers = 2  # Number of remote trainers
    
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Initialize Ray and create remote trainers
    ray.init(f"ray://192.168.1.217:10001")

    trainers = [RemoteTrainer.remote() for _ in range(num_trainers)]
    
    # Initialize all trainers with the same model
    initial_model = Net()
    initial_state = initial_model.state_dict()
    ray.get([trainer.set_model_state.remote(initial_state) for trainer in trainers])

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}")
        
        # Train on all remote trainers in parallel
        futures = [trainer.train_epoch.remote(epoch, train_loader) for trainer in trainers]
        model_states = ray.get(futures)
        
        # Average model parameters
        averaged_state = average_model_states(model_states)
        
        # Update all trainers with averaged parameters
        ray.get([trainer.set_model_state.remote(averaged_state) for trainer in trainers])
        
        # Test with averaged model
        test_model = Net().to(device)
        test_model.load_state_dict(averaged_state)
        test_accuracy = test(test_model, device, test_loader)

    # Save final model
    final_state = ray.get(trainers[0].get_model_state.remote())
    torch.save(final_state, "mnist_cnn_ray.pt")
    
    ray.shutdown()

if __name__ == '__main__':
    main()
