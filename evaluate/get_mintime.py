import os
import json
import numpy as np
import heapq

train_time_path = str
time_folder_length = 5000
pass_time = 0
k = 1
j = 0
efficientCodeTime = {}

for i in range(0, time_folder_length):
    num = str(i).zfill(4)
    times_path = train_time_path + "/" + num + "/" + "times.json"
    if os.path.exists(times_path):
        with open(times_path,'r',encoding='utf-8') as t:
            dict_data = json.load(t)
            if dict_data:
                times_list = dict_data['times']
                times_list = np.array(times_list)
                positive_numbers = [x for x in times_list if x > 0]
                min_num = min(positive_numbers) if positive_numbers else 0
                if min_num != 0:
                    efficientCodeTime[num] = min_num

filename = str
with open(filename, 'w') as f:
    json.dump(efficientCodeTime, f, indent=2)  