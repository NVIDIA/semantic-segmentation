import torch
import time
from spatialdisplconv import SpatialDisplConv

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

n = 8
h = 224
w = 224

offset = 9 # 11

#input1 = N, 3, H + 11, W + 11
#input2 = N, 11, H, W
#input3 = N, 11, H, W
#input4 = N, 2, H, W

# Note the device=cuda_device arguments here
a = torch.randn(n, 3, h + offset, w + offset, device=cuda_device, requires_grad=True).contiguous()

b = torch.randn(n, offset, h, w, device=cuda_device, requires_grad=True).contiguous()

c = torch.randn(n, offset, h, w, device=cuda_device, requires_grad=True).contiguous()

d = torch.randn(n, 2, h, w, device=cuda_device, requires_grad=True).contiguous()

sdc_layer = SpatialDisplConv(kernel_size=1).cuda()

forward = 0
backward = 0
num_runs = 100

for _ in range(num_runs):

    start = time.time()
 
    result = sdc_layer.forward(a, b, c, d)
    torch.cuda.synchronize()
    forward += time.time() - start

    sdc_layer.zero_grad()

    start = time.time()

    result_sum = result.sum()

    result_sum.backward()
    torch.cuda.synchronize()
    backward += time.time() - start

print("Forward time per iteration: %.4f ms" % (forward * 1.0 / num_runs * 1000))
print("Backward time per iteration: %.4f ms" % (forward * 1.0 / num_runs * 1000))
