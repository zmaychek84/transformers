import os
from collections import defaultdict
import pandas as pd
import sys

stage_times = defaultdict(list)
stage_times_token = defaultdict(list)
f = open(sys.argv[1])
lines = f.readlines()
cnt = 0
result = []
result_decoders = []
block = []
titles = []
record = 0
title = 0
block_size = 0
decoder_cnt = 0
for line in lines:
    # print(line)
    if "******************************************************" in line:
        record = record + 1
        title = title + 1
        continue
    else:
        if ":" not in line:
            if "csv" in line:
                break
            else:
                continue
        if "No" in line or "Model" in line:
            continue
        if ":" in line:
            if "csv" in line:
                break
    if record > 0:
        index = line.find(":")

        # if title == 1:
        name = line[:index]
        if title < 33:
            stage_times[name].append(float(line.replace("\n", "")[index + 2 :]))
        else:
            stage_times_token[name].append(float(line.replace("\n", "")[index + 2 :]))
        if title == 1:
            titles.append(name)

        block.append(line.replace("\n", "")[index + 2 :])
    if "output:" in line:
        if len(block):
            # if len(block) != block_size:
            #    title = 1
            # if title == 0:
            #    #block_size = len(block)
            #    result.append(titles)
            #    title = 1
            if title % 32:
                block.append(0)
            result_decoders.append(block)
            block = []
            # titles = []
    if "lm_head" in line:
        # print(title)
        if title == 32:
            # print(titles)
            result.append(titles)
        for item in result_decoders:
            result.append(item)

# print(stage_times)
name = ["index"]
test = pd.DataFrame(data=result)
test.to_csv("ops_0.csv", encoding="utf-8")
s = test.T
s.to_csv("ops.csv", encoding="utf-8")
average_times = {}
for stage, times in stage_times.items():
    # print(len(times))
    average_times[stage] = sum(times) / len(times)

print("---------------result ttft -----------")
# Print results
for stage, avg_time in average_times.items():
    print(f"{stage}: {avg_time:.9f}")
average_times = {}
for stage, times in stage_times_token.items():
    average_times[stage] = sum(times) / len(times)

print("---------------result token-----------")
# Print results
for stage, avg_time in average_times.items():
    print(f"{stage}: {avg_time:.9f}")
