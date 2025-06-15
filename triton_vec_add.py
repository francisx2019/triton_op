import torch
import time
import triton
import triton.language as tl

N = 98432
# DEVICE = triton.runtime.driver.active.get_active_torch_device()  # Specify the device to use

# 使用triton实现vector addition
@triton.jit
def vec_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the index of the current thread
    pid = tl.program_id(axis=0)
    # Calculate the start index for this thread
    start_index = pid * BLOCK_SIZE
    # Create a mask to handle the case where n_elements is not a multiple of BLOCK_SIZE
    mask = start_index + tl.arange(0, BLOCK_SIZE) < n_elements

    # Load elements from a and b
    a = tl.load(a_ptr + start_index + tl.arange(0, BLOCK_SIZE), mask=mask)
    b = tl.load(b_ptr + start_index + tl.arange(0, BLOCK_SIZE), mask=mask)

    # Perform vector addition
    c = a + b

    # Store the result in c
    tl.store(c_ptr + start_index + tl.arange(0, BLOCK_SIZE), c, mask=mask)

def vec_add(a, b):
    # Allocate output tensor
    c = torch.empty_like(a).to("cuda")
    # assert a.device == DEVICE and b.device == DEVICE and c.device == DEVICE
    n_elements = a.numel()

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    vec_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
    return c


a = torch.randn(N, dtype=torch.float32, device='cuda')
b = torch.randn(N, dtype=torch.float32, device='cuda')
# Vector addition using Triton
torch.cuda.synchronize()  # Ensure all previous operations are complete
time_start = time.time()
c = vec_add(a, b)
torch.cuda.synchronize()  # Ensure the addition is complete
# Print the time taken for vector addition
print(f"Time taken for vector addition: {time.time() - time_start:.6f} seconds")
# Verify the result
assert torch.allclose(c, a + b), "The result is incorrect!"