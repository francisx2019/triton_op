import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    
    """