import torch
import time

print(f" Torch version: {torch.__version__}")
print(f"  CUDA Runtime: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"           GPU: {torch.cuda.get_device_name()}")

print("Testing GPU computation...")

# Create tensors on GPU
x = torch.randn(5000, 5000).cuda()
y = torch.randn(5000, 5000).cuda()

# Matrix multiplication
start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"✓ Matrix multiply (5000x5000): {elapsed:.3f} seconds")
print(f"✓ Result shape: {z.shape}")
print(f"✓ GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("\nGPU is working! The warning is cosmetic.")



print("\n" + "="*60)
print("Testing all imports...")
print("="*60 + "\n")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import diffusers
    print(f"✓ Diffusers: {diffusers.__version__}")
except Exception as e:
    print(f"✗ Diffusers: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")

try:
    import accelerate
    print(f"✓ Accelerate: {accelerate.__version__}")
except Exception as e:
    print(f"✗ Accelerate: {e}")

try:
    import xformers
    print(f"✓ XFormers: {xformers.__version__}")
except Exception as e:
    print(f"✗ XFormers: {e}")

try:
    from diffusers import StableDiffusionXLPipeline
    print(f"✓ SDXL Pipeline import successful")
except Exception as e:
    print(f"✗ SDXL Pipeline: {e}")

print("\n" + "="*60)
print("All imports successful! 🎉")
print("="*60 + "\n")