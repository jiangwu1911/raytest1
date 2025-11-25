import ray

ray.init(f"ray://192.168.1.217:10001")

print("Connected via Ray Client")

@ray.remote(num_gpus=0.1)
def check_gpu():
    import torch
    return {
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'
    }

result = ray.get(check_gpu.remote())
print("GPU check:", result)
