import triton
import torch


def count_elapsed_time_torch(func, **kwargs):
    """
    Measure the elapsed time of a PyTorch function.

    Args:
        func (callable): The function to measure.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        float: The elapsed time in seconds.
    """
    torch.cuda.synchronize()  # Ensure all previous operations are complete
    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)

    time_start.record()
    result = func(**kwargs)
    time_end.record()

    torch.cuda.synchronize()  # Wait for the events to complete
    elapsed_time = time_start.elapsed_time(time_end) / 1000.0  # Convert ms to seconds

    return elapsed_time, result


# 实现triton的benchmark函数
# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['size'],
#         x_vals=[2**i for i in range(10, 20)],
#         line_arg='op',
#         line_vals=['triton', 'torch'],
#         styles=[('triton', {'marker': 'o', 'color': 'blue'}),
#                 ('torch', {'marker': 'x', 'color': 'red'})],
#         ylabel='GB/s', # label name for the y-axis
#         plot_name='Triton vs Torch Performance',
#     ))
# def benchmark(op, size):
#     """
#     Benchmark function for Triton and Torch operations.
#     Args:
#         op (str): The operation to benchmark ('triton' or 'torch').
#         size (int): The size of the input tensors.
#     """
#     a = torch.randn(size, dtype=torch.float16, device='cuda')
#     b = torch.randn(size, dtype=torch.float16, device='cuda')

#     if op == 'triton':
#         # Call the Triton vector addition function
#         elapsed_time, _ = count_elapsed_time_torch(vec_add, a=a, b=b)
#     elif op == 'torch':
#         # Call the PyTorch vector addition function
#         elapsed_time, _ = count_elapsed_time_torch(torch_vec_add, a=a, b=b)
#     else:
#         raise ValueError(f"Unknown operation: {op}")

#     # Calculate the throughput in GB/s
#     throughput = (2 * size * a.element_size()) / (elapsed_time * 1e9)
