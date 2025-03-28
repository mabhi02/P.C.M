# check_cuda.py

import torch

def check_cuda():
    print("CUDA Available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print("Name:", torch.cuda.get_device_name(i))
            print("Memory Allocated:", torch.cuda.memory_allocated(i) // (1024**2), "MB")
            print("Memory Cached:", torch.cuda.memory_reserved(i) // (1024**2), "MB")
            print("Compute Capability:", torch.cuda.get_device_capability(i))
    else:
        print("No CUDA-compatible GPU found.")

if __name__ == "__main__":
    check_cuda()
