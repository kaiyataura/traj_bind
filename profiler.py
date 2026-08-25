import sys
import torch
import time
from collections import defaultdict
from model import TeacherModel

device = 'mps'
dtype = torch.float32
model = TeacherModel().to(device).to(dtype).train()

B, T, N = 4, 16, 250
R = 10

coords = torch.randn(B, T, N, 4, 3, device=device, dtype=dtype)
residues = torch.randint(0, 20, (B, N), device=device)
props = torch.randn(B, N, 5, device=device, dtype=dtype)
masks = torch.ones(B, N, device=device, dtype=torch.bool)
core_masks = torch.ones(B, N, device=device, dtype=torch.bool)

for _ in range(2):
    model(coords, residues, props, masks, core_masks, mode='train')
    if device == 'mps': torch.mps.empty_cache()
torch.mps.synchronize()

line_times = defaultdict(float)
last_time = None
last_line = None

def trace_lines(frame, event, arg):
    global last_time, last_line
    if event != 'line': return trace_lines
    if 'model.py' not in frame.f_code.co_filename: return trace_lines
        
    torch.mps.synchronize()
    now = time.perf_counter()
    
    if last_line is not None: line_times[last_line] += now - last_time
    last_line = frame.f_lineno
    last_time = now
    
    return trace_lines

print("Profiling lines...")
torch.mps.synchronize()
last_time = time.perf_counter()
sys.settrace(trace_lines)
for _ in range(R):
    model(coords, residues, props, masks, core_masks, mode='train')
    if device == 'mps': torch.mps.empty_cache()
sys.settrace(None)
torch.mps.synchronize()
if last_line is not None: line_times[last_line] += time.perf_counter() - last_time

with open('model.py', 'r') as f: source_lines = f.readlines()

print("\n" + "=" * 120)
print(f" {'Line':<5} | {'Time (ms)':>9} | {'Source Code'}")
print("-" * 120)
sorted_lines = sorted(line_times.items(), key=lambda x: x[1], reverse=True)
for lineno, t in sorted_lines[:30]:
    if lineno - 1 < len(source_lines):
        print(f" {lineno:<5} | {t / R * 1000:>9.2f} | {source_lines[lineno - 1].strip()[:80]}")
print("=" * 120)
print(f" {'Total':<5} | {sum(line_times.values()) / R * 1000:>9.2f} | ")
