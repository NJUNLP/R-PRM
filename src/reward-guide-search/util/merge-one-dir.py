import os
import sys
import json

dir_path = sys.argv[1]
filenames = os.listdir(dir_path)

filenames = [filename for filename in filenames if filename.endswith(".json") and filename.startswith("result")]
filenames.sort(key=lambda x: int(x.split("-")[1]))

new_dataset = []
for filename in filenames:
    with open(os.path.join(dir_path, filename), 'r') as f:
        data = json.load(f)
        new_dataset += data
dataset = []
for item in new_dataset:
    steps = item.get("final_steps", [])
    if len(steps) == 0:
        steps = item.get("previous_steps", [])
    item = {
        "id": len(dataset),
        "response": ["\n".join(steps)],
    }
    dataset.append(item)
with open(os.path.join(dir_path, "response.jsonl"), 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")