import json
import sys
import os

dir_path = sys.argv[1]

filenames = os.listdir(dir_path)
# 保留所有满足result-*-*.json的文件
filenames = [filename for filename in filenames if filename.startswith("result-") and filename.endswith(".json")]

filenames.sort(key=lambda x: int(x.split('-')[1]))

dataset = []
for filename in filenames:
    with open(os.path.join(dir_path, filename), 'r') as f:
        data = json.load(f)
        dataset += data

new_dataset = []
for data in dataset:
    target_stes = data.get("previous_steps", [])
    if len(target_stes) == 0:
        target_stes = data.get("final_steps", [])
    item = {
        "id": len(new_dataset),
        "response": ["\n".join(target_stes)]
    }
    new_dataset.append(item)

with open(os.path.join(dir_path, "response.jsonl"), 'w') as f:
    for item in new_dataset:
        f.write(json.dumps(item) + '\n')