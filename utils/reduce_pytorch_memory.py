"""
Workaround to reduce PyTorch memory usage during import.
This can help when paging file is limited.
"""

import os

# Set environment variables before importing torch
# These can reduce initial memory footprint
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'  # Reduce thread overhead
os.environ['MKL_NUM_THREADS'] = '1'

# Try to use CPU-only PyTorch if available (uses less memory)
# Uncomment if you have CPU-only version:
# os.environ['TORCH_DEVICE'] = 'cpu'

# Disable some memory-intensive features
os.environ['TORCH_USE_CUDA_DSA'] = '0'  # Disable CUDA device-side assertions

print("PyTorch memory reduction settings applied")
print("These may help with paging file issues")



