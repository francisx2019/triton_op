import torch

# Vector addition using PyTorch
def torch_vec_add(a, b):
    """
    Perform vector addition using PyTorch.
    
    Args:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor.
    
    Returns:
        torch.Tensor: Result of the vector addition.
    """
    return torch.add(a, b)

