import os
import json
import numpy as np
import heapq

generate_code_time = str
generate_code_time_with_mask = str
time_folder_length = 5000
j = 0
left = 0
right = 0
fasterTime = {}
solwTime = {}

for i in range(0, time_folder_length):
    num = str(i).zfill(4)
    if os.path.exists(generate_code_time) & os.path.exists(generate_code_time_with_mask):
        with open(generate_code_time,'r',encoding='utf-8') as gct:
            code_times = json.load(gct)
        with open(generate_code_time_with_mask,'r',encoding='utf-8') as gcwcst:
            code_with_mask_times = json.load(gcwcst)
    if (num in code_times) & (num in code_with_mask_times):
        j += 1
        if code_times[num] < code_with_mask_times[num]:
            right += 1
            rightrate = code_times[num] / code_with_mask_times[num]
            solwTime[num] = [code_times[num], code_with_mask_times[num], rightrate]
        if code_times[num] >= code_with_mask_times[num]:
            left += 1
            leftrate = code_times[num] / code_with_mask_times[num]
            fasterTime[num] = [code_times[num], code_with_mask_times[num], leftrate]

filename1 = str
with open(filename1, 'w') as f:
    json.dump(solwTime, f, indent=2)  

filename2 = str
with open(filename1, 'w') as f:
    json.dump(fasterTime, f, indent=2)  
