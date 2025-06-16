
import argparse
import torch
from add.triton_vec_add import vec_add
from add.torch_vec_add import torch_vec_add
from softmax.torch_softmax import torch_softmax
from utils import count_elapsed_time_torch 

def main():
    parser = argparse.ArgumentParser(description='Run a benchmark test for a given operator.')
    parser.add_argument("-op", '--operator', type=str, default="vec_add", help='The operator to benchmark.')
    args = parser.parse_args()

    if args.operator == 'vec_add':
        N = 8192  # Size of the vectors
        torch.cuda.manual_seed(0)  # For reproducibility
        a = torch.randn(N, dtype=torch.float16, device='cuda')
        b = torch.randn(N, dtype=torch.float16, device='cuda')
        els1, out1 = count_elapsed_time_torch(vec_add, a=a, b=b)  # Measure the time taken for vector addition
        els2, out2 = count_elapsed_time_torch(torch_vec_add, a=a, b=b)  # Measure the time taken for vector addition
        # Verify the result
        assert torch.allclose(out1, out2), "The result is incorrect!"
        print(f"Time taken for triton vector addition: {els1:.6f} seconds")
        print(f"Time taken for torch vector addition: {els2:.6f} seconds")
    elif args.operator == 'softmax':
        pass
    else:
        raise ValueError(f"Unknown operator: {args.operator}.")

if __name__ == "__main__":
    main()
