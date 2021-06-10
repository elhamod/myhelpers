import torch 

def get_cuda_memory(device):
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device) 
    a = torch.cuda.memory_allocated(device)
    return r-a