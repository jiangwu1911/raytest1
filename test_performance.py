import ray
import time
import numpy as np

ray.init("ray://192.168.1.217:10001")

@ray.remote(num_gpus=0.1)
class LargePerformanceTest:
    def __init__(self):
        import torch
        self.device = torch.device("cuda")
    
    def large_matrix_multiplication(self, size=1024, iterations=50):
        """Test large matrix multiplication"""
        import torch
        
        print(f"Starting large matrix multiplication: {size}x{size}, {iterations} iterations")
        
        # Create large random matrices
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        
        # Warm up
        for _ in range(5):
            _ = torch.matmul(a, b)
        
        # Time the operation
        start_time = time.time()
        for i in range(iterations):
            result = torch.matmul(a, b)
            if i % 10 == 0:
                print(f"  Iteration {i}/{iterations}")
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # Calculate performance
        avg_time = total_time / iterations
        gflops = (2 * size ** 3) / (avg_time * 1e9)
        
        return {
            'matrix_size': size,
            'iterations': iterations,
            'total_time': total_time,
            'average_time_ms': avg_time * 1000,
            'gflops': gflops,
            'result_norm': torch.norm(result).item()
        }
    
    def deep_learning_forward_pass(self, batch_size=8, image_size=64, channels=3, layers=20):
        """Simulate deep learning forward pass"""
        import torch
        import torch.nn as nn
        
        print(f"Simulating deep learning: batch_size={batch_size}, layers={layers}")
        
        # Create simulated deep learning model with smaller channels
        class DeepModel(nn.Module):
            def __init__(self, layers):
                super(DeepModel, self).__init__()
                self.layers = nn.ModuleList([
                    nn.Conv2d(channels if i == 0 else 32, 32, 3, padding=1)
                    for i in range(layers)
                ])
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(32, 10)
            
            def forward(self, x):
                for layer in self.layers:
                    x = self.relu(layer(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = DeepModel(layers).to(self.device)
        model.eval()
        
        # Create input data
        input_data = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        
        # Warm up
        for _ in range(5):
            _ = model(input_data)
        
        # Time the operation
        start_time = time.time()
        iterations = 30
        for i in range(iterations):
            output = model(input_data)
            if i % 10 == 0:
                print(f"  Forward pass {i}/{iterations}")
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        return {
            'batch_size': batch_size,
            'image_size': image_size,
            'layers': layers,
            'iterations': iterations,
            'total_time': total_time,
            'time_per_iteration_ms': (total_time / iterations) * 1000,
            'output_shape': list(output.shape)
        }
    
    def memory_bandwidth_test(self, size_gb=0.25):
        """Test GPU memory bandwidth"""
        import torch
        
        print(f"Testing memory bandwidth with {size_gb}GB data")
        
        # Calculate number of elements (float32)
        size_elements = int(size_gb * 1024 * 1024 * 1024 / 4)
        
        # Create large tensors
        a = torch.randn(size_elements, device=self.device)
        b = torch.randn(size_elements, device=self.device)
        
        # Test memory copy bandwidth
        start_time = time.time()
        iterations = 10
        for i in range(iterations):
            c = a + b
            if i % 5 == 0:
                print(f"  Bandwidth test iteration {i}/{iterations}")
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # Calculate bandwidth
        data_transferred = size_gb * 2 * iterations
        bandwidth = data_transferred / total_time
        
        return {
            'data_size_gb': size_gb,
            'iterations': iterations,
            'total_time': total_time,
            'bandwidth_gb_s': bandwidth,
            'result_sum': c.sum().item()
        }

print("Starting large performance tests...")

perf_test = LargePerformanceTest.remote()

# Test 1: Large matrix multiplication
print("\n" + "="*50)
print("TEST 1: Large Matrix Multiplication")
print("="*50)
matmul_result = ray.get(perf_test.large_matrix_multiplication.remote(1024, 30))
print("\nMatrix Multiplication Results:")
print(f"Matrix Size: {matmul_result['matrix_size']}x{matmul_result['matrix_size']}")
print(f"Iterations: {matmul_result['iterations']}")
print(f"Total Time: {matmul_result['total_time']:.2f}s")
print(f"Average Time: {matmul_result['average_time_ms']:.2f}ms")
print(f"Performance: {matmul_result['gflops']:.2f} GFLOPs")

# Test 2: Deep learning simulation
print("\n" + "="*50)
print("TEST 2: Deep Learning Forward Pass")
print("="*50)
dl_result = ray.get(perf_test.deep_learning_forward_pass.remote(8, 64, 3, 20))
print("\nDeep Learning Results:")
print(f"Batch Size: {dl_result['batch_size']}")
print(f"Layers: {dl_result['layers']}")
print(f"Iterations: {dl_result['iterations']}")
print(f"Total Time: {dl_result['total_time']:.2f}s")
print(f"Time per Iteration: {dl_result['time_per_iteration_ms']:.2f}ms")
print(f"Output Shape: {dl_result['output_shape']}")

# Test 3: Memory bandwidth
print("\n" + "="*50)
print("TEST 3: Memory Bandwidth Test")
print("="*50)
bandwidth_result = ray.get(perf_test.memory_bandwidth_test.remote(0.25))
print("\nMemory Bandwidth Results:")
print(f"Data Size: {bandwidth_result['data_size_gb']}GB")
print(f"Iterations: {bandwidth_result['iterations']}")
print(f"Total Time: {bandwidth_result['total_time']:.2f}s")
print(f"Bandwidth: {bandwidth_result['bandwidth_gb_s']:.2f} GB/s")

print("\nAll performance tests completed!")
