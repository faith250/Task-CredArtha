import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA not available. Checking potential issues...")
    
    # Check if GPU is detected by the system
    try:
        import subprocess
        gpu_info = subprocess.check_output('nvidia-smi', shell=True)
        print("nvidia-smi output:")
        print(gpu_info.decode('utf-8'))
        print("GPU is detected by system but not by PyTorch. This suggests a PyTorch/CUDA version mismatch.")
    except Exception as e:
        print(f"nvidia-smi failed: {e}")
        print("GPU might not be properly installed or drivers are missing.")