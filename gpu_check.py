import torch

def check_gpu_characteristics():
    print("=" * 50)
    print("PyTorch GPU Diagnosis Script")
    print("=" * 50)
    
    # 1. Check if CUDA (GPU support) is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n[!] No GPU detected by PyTorch. Running on CPU.")
        print("Check if your GPU drivers and PyTorch CUDA version match.")
        print("=" * 50)
        return

    # 2. Get the number of available GPUs
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs detected: {device_count}")
    print("-" * 50)

    # 3. Iterate through each GPU and extract characteristics
    for i in range(device_count):
        print(f"--- GPU Device {i} ---")
        
        # Device Name
        device_name = torch.cuda.get_device_name(i)
        print(f"Name:                  {device_name}")
        
        # Compute Capability (Architectural version)
        major, minor = torch.cuda.get_device_capability(i)
        print(f"Compute Capability:    {major}.{minor}")
        
        # Memory Characteristics
        # Note: PyTorch returns memory in bytes, converting to Gigabytes (GB)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"Total Memory:          {total_memory:.2f} GB")
        
        # Current Memory Allocation
        allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)
        cached_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)
        print(f"Currently Allocated:   {allocated_memory:.2f} GB")
        
        # Check if it's the current default device
        is_current = "Yes" if i == torch.cuda.current_device() else "No"
        print(f"Default Active Device: {is_current}")
        print("-" * 50)

    print("=" * 50)

if __name__ == "__main__":
    check_gpu_characteristics()