import torch
from torch.autograd import profiler

class Basic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

f = Basic()
example_args = (torch.randn(3, 3), torch.randn(3, 3))

# Use profiler to capture the autograd graph
with profiler.profile() as prof:
    output = f(*example_args)

print(output)

# Print the captured autograd graph
print(prof.key_averages().table(sort_by="cpu_time_total"))
