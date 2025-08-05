import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# PERF: Helper to detect optimal number of DataLoader workers
def get_optimal_workers():
    cpu = os.cpu_count() or 4
    return min(cpu, 8)  # Fallback to 4 if detection fails

# PERF: Auto-adjust batch size based on GPU memory headroom with conservative 25% increments
def auto_adjust_batch_size(initial_batch_size, headroom_threshold=4 * 1024 * 1024 * 1024):  # 4 GB headroom
    if not torch.cuda.is_available():
        return initial_batch_size
    
    free_mem, total_mem = torch.cuda.mem_get_info()
    # Use more conservative incremental growth to avoid OOM
    if free_mem > headroom_threshold:
        return int(initial_batch_size * 1.25)  # 25% increase instead of doubling
    return initial_batch_size

# PERF: Setup profiler for a given number of steps
import datetime

def setup_profiler(steps=50, output_file=None):
    if output_file is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'runs/profile_{ts}.json'
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_file),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )