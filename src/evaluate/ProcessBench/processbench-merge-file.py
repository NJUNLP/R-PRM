from datasets import load_dataset
import json
from tqdm import tqdm
import os
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import sys

dir_path = sys.argv[1] # 例如："/home/adminad/MCTS-RM/GenRM/evaluate/Llemma/output/llemma-7b-prm-metamath-level-1to3-hf/ProcessBench"
dirnames = os.listdir(dir_path)

for dirname in dirnames:
    now_dir = os.path.join(dir_path, dirname)
    filenames = os.listdir(now_dir)
    filenames.sort(key=lambda x: int(x.split("_")[1]))
    dataset = []
    for filename in filenames:
        with open(os.path.join(now_dir, filename), "r") as f:
            data = json.load(f)
            dataset += data
    new_dataset = []
    for data in dataset:
        item = {
            "id": len(new_dataset),
            "query": data["question"],
            "steps": data["responses"][0]["steps"],
            "math-shepherd-scores": data["responses"][0]["steps_rewards"]
        }
        new_dataset.append(item)
    with open(os.path.join(now_dir, "response.jsonl"), "w") as f:
        for data in new_dataset:
            f.write(json.dumps(data) + "\n")
