import ray
import numpy as np

# Connect to remote Ray cluster
ray.init("ray://192.168.1.217:10001")
print("Connected to remote GPU cluster")

@ray.remote(num_gpus=0.5)
class GPUOperations:
    def __init__(self):
        import torch
        self.device = torch.device("cuda")
        print(f"GPU worker initialized on: {self.device}")
    
    def tensor_operations(self, data):
        """Test basic tensor operations on GPU"""
        import torch
        # Move data to GPU
        tensor = torch.tensor(data, device=self.device).float()
        
        # Perform operations on GPU
        result1 = tensor * 2.0
        result2 = torch.sin(tensor)
        result3 = torch.matmul(tensor, tensor.T)  # Matrix multiplication
        
        return {
            'original': tensor.cpu().numpy(),
            'multiplied': result1.cpu().numpy(),
            'sine': result2.cpu().numpy(),
            'matmul': result3.cpu().numpy()
        }
    
    def memory_test(self, size_mb=100):
        """Test GPU memory allocation"""
        import torch
        size_bytes = size_mb * 1024 * 1024
        elements = size_bytes // 4  # float32 is 4 bytes
        
        try:
            # Allocate memory on GPU
            gpu_tensor = torch.zeros(elements, dtype=torch.float32, device=self.device)
            
            # Perform some computation
            result = torch.sum(gpu_tensor) + 1.0
            
            return f"Successfully allocated {size_mb}MB on GPU, result: {result.item()}"
        except Exception as e:
            return f"Memory allocation failed: {e}"

# Create GPU worker
print("Creating remote GPU worker...")
gpu_worker = GPUOperations.remote()

# Test 1: Basic tensor operations
print("\n1. Testing basic tensor operations...")
data = np.random.rand(5, 5).astype(np.float32)
result = ray.get(gpu_worker.tensor_operations.remote(data))
print("Tensor operations completed successfully")
print(f"Original shape: {result['original'].shape}")
print(f"Matrix multiplication shape: {result['matmul'].shape}")

# Test 2: Memory allocation
print("\n2. Testing GPU memory allocation...")
memory_result = ray.get(gpu_worker.memory_test.remote(50))
print("Memory test:", memory_result)

print("\nAll tests completed! Remote GPU is working correctly.")
