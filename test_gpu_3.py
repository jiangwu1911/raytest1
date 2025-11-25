import ray
import numpy as np

ray.init("ray://192.168.1.217:10001")

@ray.remote(num_gpus=0.5)
class NeuralNetworkTest:
    def __init__(self):
        import torch
        import torch.nn as nn
        
        # Simple neural network for testing
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        self.device = torch.device("cuda")
        self.model = SimpleNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        print("Neural network initialized on GPU")
    
    def train_step(self, x_data, y_data):
        """Perform one training step on GPU"""
        import torch
        
        # Move data to GPU
        x_tensor = torch.tensor(x_data, device=self.device).float()
        y_tensor = torch.tensor(y_data, device=self.device).float()
        
        # Training step
        self.optimizer.zero_grad()
        predictions = self.model(x_tensor)
        loss = self.criterion(predictions, y_tensor)
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'predictions': predictions.detach().cpu().numpy()
        }
    
    def inference(self, x_data):
        """Perform inference on GPU"""
        import torch
        with torch.no_grad():
            x_tensor = torch.tensor(x_data, device=self.device).float()
            predictions = self.model(x_tensor)
            return predictions.cpu().numpy()

print("Testing neural network on remote GPU...")

# Create neural network worker
nn_worker = NeuralNetworkTest.remote()

# Generate dummy data
x_data = np.random.rand(32, 10).astype(np.float32)  # 32 samples, 10 features
y_data = np.random.rand(32, 1).astype(np.float32)   # 32 targets

# Test training
print("Performing training step...")
training_result = ray.get(nn_worker.train_step.remote(x_data, y_data))
print(f"Training loss: {training_result['loss']:.4f}")
print(f"Predictions shape: {training_result['predictions'].shape}")

# Test inference
print("\nPerforming inference...")
inference_result = ray.get(nn_worker.inference.remote(x_data[:5]))  # First 5 samples
print(f"Inference result shape: {inference_result.shape}")
print(f"Inference results: {inference_result.flatten()}")

print("\nNeural network tests completed successfully!")
