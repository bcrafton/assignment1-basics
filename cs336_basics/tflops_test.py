
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import time

'''
N = 16384
x = torch.normal(mean=0, std=1, size=(N, N)).to(torch.float16).to('cuda')
w = torch.normal(mean=0, std=1, size=(N, N)).to(torch.float16).to('cuda')

# with_flops just counts the number of flops in the operation itself. it will always print '8.796' TFLOPs for N=16384
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True, with_flops=True) as prof:
  # The operation to profile
  torch.matmul(x, w) 

# Print the profiling results, sorted by total time
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# Export for a timeline trace view in Perfetto or TensorBoard
prof.export_chrome_trace("matrix_multiplication_trace.json")

# maybe check this link out?
# https://medium.com/@michael.diggin/the-power-of-8-getting-the-most-out-of-tensor-cores-c7704ae0c5c1

print (prof.key_averages())
'''

N = 16384*2*2
x = torch.normal(mean=0, std=1, size=(N, N)).to('cuda')
w = torch.normal(mean=0, std=1, size=(N, N)).to('cuda')

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
  # The operation to profile
  torch.matmul(x, w) 

# Print the profiling results, sorted by total time
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# Export for a timeline trace view in Perfetto or TensorBoard
prof.export_chrome_trace("matrix_multiplication_trace.json")

# maybe check this link out?
# https://medium.com/@michael.diggin/the-power-of-8-getting-the-most-out-of-tensor-cores-c7704ae0c5c1

