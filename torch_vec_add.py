import torch
import time

N = 98432

a = torch.randn(N, dtype=torch.float32, device='cuda')
b = torch.randn(N, dtype=torch.float32, device='cuda')
c = torch.empty(N, dtype=torch.float32, device='cuda')
# Vector addition using PyTorch
torch.cuda.synchronize()  # Ensure all previous operations are complete
time_start = time.time()
c = torch.add(a, b)
torch.cuda.synchronize()  # Ensure the addition is complete
print(f"Time taken for vector addition: {time.time() - time_start:.6f} seconds")
# Verify the result
assert torch.allclose(c, a + b), "The result is incorrect!"
