import os
import sys
import json
from datasets import load_dataset
import numpy as np

def get_first_number(numbers):
    for i, number in enumerate(numbers):
        if number < 0.5:
            return i
    return -1

origin_dataset_path = sys.argv[1]
response_dataset_path = sys.argv[2]
output_path = os.path.join(os.path.dirname(response_dataset_path), "final_result.json")

id_map = {}
origin_dataset = load_dataset(
    "json",
    data_files=origin_dataset_path,
    split="train",
)
response_dataset = load_dataset(
    "json",
    data_files=response_dataset_path,
    split="train",
)

id_map = {}
for i, item in enumerate(response_dataset):
    id_map[item["id"]] = i
new_dataset = []
for data in origin_dataset:
    item = data
    idx = id_map.get(data["id"], -1)
    if idx == -1:
        assert False
    item["yes_probabilities"] = response_dataset[idx]["yes_probabilities"]
    item["yes_probability_average"] = response_dataset[idx]["yes_probability_average"]
    item["PRM-CoT"] = response_dataset[idx]["PRM-CoT"][0]
    new_dataset.append(item)

# 按照origin_id进行聚类
origin_id_map = {}
for i, item in enumerate(new_dataset):
    origin_id = item["origin_id"]
    if origin_id not in origin_id_map:
        origin_id_map[origin_id] = []
    origin_id_map[origin_id].append(item)

new_dataset = []
for origin_id, items in origin_id_map.items():
    items.sort(key=lambda x: x["id"])
    yes_probability_average = [item["yes_probability_average"] for item in items]
    now_item = {
        "id": origin_id,
        "question": items[0]["question"],
        "steps": [item["now_step"] for item in items],
        "yes_probability_average": yes_probability_average,
        "label": items[0]["label"],
        "predict": get_first_number(yes_probability_average),
    }
    now_item["match"] = now_item["label"] == now_item["predict"]
    new_dataset.append(now_item)

with open(output_path, "w") as f:
    json.dump(new_dataset, f, indent=4)

bench_names = ["gsm8k", "math", "olympiadbench", "omnimath"]
final_dataset = []
for bench_name in bench_names:
    now_dataset = [item for item in new_dataset if item["id"].startswith(bench_name)]
    if not now_dataset:
        continue
    correct_data = [item for item in now_dataset if item["label"] == -1]
    error_data = [item for item in now_dataset if item["label"] != -1]

    acc1 = np.mean([e['match'] for e in error_data]) * 100 if error_data else 0
    acc2 = np.mean([e['match'] for e in correct_data]) * 100 if correct_data else 0
    
    # 计算f1分数
    if acc1 + acc2 == 0:
        f1 = 0
    else:
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    print(f"{bench_name}: error accuracy={acc1:.2f}, correct accuracy={acc2:.2f}, f1={f1:.2f}")
    result = {
        'bench_name': bench_name,
        'error_acc': acc1,
        'correct_acc': acc2,
        'f1': f1
    }
    final_dataset.append(result)

result_path = os.path.join(os.path.dirname(response_dataset_path), "result-score.json")
with open(result_path, "w") as f:
    json.dump(final_dataset, f, indent=4)

