import torch

def torch_softmax(x, dim=None, dtype=None):
    """
    Compute the softmax of a tensor using PyTorch.
    
    Args:
        x (torch.Tensor): Input tensor.
        dim (int, optional): Dimension along which to compute softmax. Default is None.
        dtype (torch.dtype, optional): Data type of the output tensor. Default is None.
    
    Returns:
        torch.Tensor: Softmax of the input tensor.
    """
    return torch.nn.functional.softmax(x, dim=dim, dtype=dtype)
