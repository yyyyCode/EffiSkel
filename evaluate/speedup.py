import os
import json

fastertime_path = str
slowertime_path = str
time_folder_length = 5000
speedup = 0.0                 
avg_speedup = 0.0             
fasterspeedup_total = 0.0     
slowerspeedup_total = 0.0     
fastertime_num = 0          
slowertime_num = 0          
num = 0                     
rate = 0                   

for i in range(0, time_folder_length):
    num = str(i).zfill(4)
    if os.path.exists(fastertime_path):
        with open(fastertime_path,'r',encoding='utf-8') as fp:
            code_fastertimes = json.load(fp)
    if os.path.exists(slowertime_path):
        with open(slowertime_path,'r',encoding='utf-8') as sp:
            code_slowertimes = json.load(sp)
    if (num in code_fastertimes):
        time = code_fastertimes[num]
        fastertime_num += 1
        fasterspeedup_total += time[2]
    if (num in code_slowertimes):
        time = code_slowertimes[num]
        slowertime_num += 1
        slowerspeedup_total += time[2]
speedup = fasterspeedup_total + slowerspeedup_total
num = fastertime_num + slowertime_num
avg_speedup = speedup / num
rate = fastertime_num / (fastertime_num + slowertime_num)
print(fasterspeedup_total)
print(slowerspeedup_total)
print(speedup)
print(avg_speedup)
print(fastertime_num)
print(slowertime_num)
print(num)
print(rate)
