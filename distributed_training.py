import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    pass


def init_process(rank, size, fn, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank = rank, world_size= size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



# blocking point to point communication
# def run(rank, size):
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         dist.send(tensor = tensor, dst = 1)

#     else:
#         dist.recv(tensor= tensor, src = 0)
    
#     print("Rank", rank, "has data", tensor[0])




# all reduce example
def run(rank, size):
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group = group)
    print("Rank", rank,  " has data ", tensor[0])





class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index


    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes= [0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)


        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


# parititoning mnist
def partition_dataset():
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081,)),
    ]))
    size = dist.get_world_size()
    bsz = 128 /float(size)
    partition_sizes = [1.0 /size for _ in range(size)]
    paritition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size = bsz, shuffle=True)
    return train_set, bsz


    
def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.5)

    num_batches = ceil(len(train_dataset) / float(bsz))

    for epoch in range(10):
        epoch_loss = 0.0
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        epoch_loss += loss.item()
        loss.backward()
        average_gradients(model)
        optimizer.step()
    print('Rank: ', dist.get_rank(), ", epoch ", epoch , ":", epoch_loss / num_batches)



def average_gradients(model):
    size = float(dist.get_world_size)
    for param in model.parameters():
        dist.all_reduce(param.get.data, op = dist.ReduceOp.SUM)
        param.grad.data /= size




#  our own ring alliance


def allreduce(send, recv):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = send.clone()
    recv_buff = send.clone()
    accum = send.clone()


    left = ((rank - 1) + size) % size
    right = ((rank + 1) + size) % size

    for i in range(size - 1):
        if i % 2 == 0:
            # send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv.buff[:]


        else:
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send_buff[:]
        send_req.wait()
    recv[:] = accum[:]

    
     
