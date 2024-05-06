import torch
from torch import Tensor
from torch.distributed import rpc
from torch.distributed.rpc import future

@torch.jit.script
def script_add(x: Tensor, y: Tensor) -> Tensor:
    return x + y


@rpc.functions.async_execution
@torch.jit.script
def async_add(to: str, x: Tensor, y:Tensor) -> Future:
    return rpc.rpc_async(to, script_add, (x, y))

# on worker 0
ret = rpc.rpc_sync(
    "worker1",
    async_add,
    args=("worker2", torch.ones(2), torch.tensor(1))
)

print(ret)