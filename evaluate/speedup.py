import os
import json
from math import floor, ceil


fastertime_path = str
slowertime_path = str
time_folder_length = 5000
fastertime_num = 0
slowertime_num = 0  


def safe_load(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

code_fastertimes = safe_load(fastertime_path)  
code_slowertimes = safe_load(slowertime_path)


all_speedups = []
for d in (code_fastertimes, code_slowertimes):
    for k, v in d.items():
        
        if isinstance(v, (list, tuple)) and len(v) >= 3 and isinstance(v[2], (int, float)):
            all_speedups.append(float(v[2]))


if not all_speedups:
    print("No speedup data found.")
    exit(0)


def percentile(values, p):
    vals = sorted(values)
    n = len(vals)
    if n == 1:
        return vals[0]
    pos = (p/100) * (n - 1)
    lo, hi = floor(pos), ceil(pos)
    if lo == hi:
        return vals[int(pos)]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

p5  = percentile(all_speedups, 5)
p95 = percentile(all_speedups, 95)


def winsorize(x, lo, hi):
    if x < lo: return lo
    if x > hi: return hi
    return x

trimmed = [x for x in all_speedups if p5 <= x <= p95]
avg_speedup = sum(trimmed) / len(trimmed) if trimmed else 0.0

print(f"P5: {p5}")
print(f"P95: {p95}")
print(f"Count(all): {len(all_speedups)}")
print(f"Count(trimmed): {len(trimmed)}")
print(f"Avg speedup (P5–P95): {avg_speedup}")