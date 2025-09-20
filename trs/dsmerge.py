#!/bin/python3
import json
import math
import os
import argparse

def collect(dir, ds_range):
    if ds_range is None:
        return json.load(open(f"{dir}/All.json"))
    elif isinstance(ds_range, int):
        data = []
        for i in range(ds_range):
            data += json.load(open(f"{dir}/{i+1}q{ds_range}.json"))
        return data
    else:
        raise ValueError("Invalid ds_range format.")

def auto_collect(dir):
    files = os.listdir(dir)
    if "All.json" in files:
        return collect(dir, None)
    else:
        ds_range = None
        for file in files:
            if file.endswith(".json"):
                ds_range = int(file.split(".")[0].split("q")[1])
                break
        if ds_range is None:
            raise ValueError("No dataset found: dir="+str(dir))
        return collect(dir, ds_range)
    
def ds_split(l:int, m:int, n:int):
    k = math.trunc(l/n)
    return k*(m-1), l if m==n else k*m

def release(dir, ds_range, data):
    if ds_range is None:
        json.dump(data, open(f"{dir}/All.json", 'w'), indent=4)
    elif isinstance(ds_range, int):
        for i in range(ds_range):
            ds = ds_split(len(data), i+1, ds_range)
            new_file = f"{dir}/{i+1}q{ds_range}.json"
            if os.path.exists(new_file):
                raise ValueError(f"{new_file} already exists.")
            json.dump(data[ds[0]:ds[1]], open(new_file, 'w'), indent=4)
    else:
        raise ValueError("Invalid ds_range format.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='Directory of dataset')
    parser.add_argument('--dst', type=int, required=False, default=None, help='Dataset range (e.g. , 3q10)')
    args = parser.parse_args()
    
    data_src = auto_collect(args.dir)
    release(args.dir, args.dst, data_src)