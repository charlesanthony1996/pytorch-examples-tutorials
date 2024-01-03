import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

checkpoint_dir = 'checkpoint'

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net3 = nn.Linear(16, 8)


    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


    
def setup(rank, world_size):
    os.environ["master_addr"] = "localhost"
    os.environ["master_port"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"running basic fsdp checkpoint saving example on rank {rank}")
    setup(rank, world_size)

    # create a model and move it to gpu with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    model(torch.rand(8, 16, device="cuda")).sum().backward()
    optimizer.step()

    FSDP.set_state_dict_type(model, StateDictType.SHARED_STATE_DICT)

    state_dict = {
        "model": model.state_dict()
    }

    DCP.save_state_dict(state_dict = state_dict, storage_writer = DCP.FileSystemWriter(checkpoint_dir))

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"running fsdp checkpoint example on {world_size} devices")
    mp.spawn(run_fsdp_checkpoint_save_example, args=(world_size), nprocs=(world_size), join=True)



    



