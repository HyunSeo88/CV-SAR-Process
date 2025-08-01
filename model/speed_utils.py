import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# PERF: Helper to detect optimal number of DataLoader workers
def get_optimal_workers():
    return os.cpu_count() * 2 if os.cpu_count() else 4  # Fallback to 4 if detection fails

# PERF: Auto-adjust batch size based on GPU memory headroom
def auto_adjust_batch_size(initial_batch_size, headroom_threshold=2 * 1024 * 1024 * 1024):  # 2 GB
    if not torch.cuda.is_available():
        return initial_batch_size
    
    free_mem, total_mem = torch.cuda.mem_get_info()
    if free_mem > headroom_threshold:
        return initial_batch_size * 2  # Double if enough headroom
    return initial_batch_size

# PERF: Setup profiler for a given number of steps
def setup_profiler(steps=50, output_file='runs/profile.json'):
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_file),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )