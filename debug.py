

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


for i in range(0,1000) :
    data = torch.randn([512, 1024, 1, 1], dtype=torch.float, device='cuda', requires_grad=True)
    net = torch.nn.Conv2d(1024, 1024, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=32)
    net = net.cuda().float()
    out = net(data)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()