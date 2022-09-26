import torch
device = torch.device('mps')
x= torch.rand ((10000,10000), dtype=torch.float32)
y= torch.rand ((10000,10000), dtype=torch.float32)
x = x. to (device)
y = y. to (device)