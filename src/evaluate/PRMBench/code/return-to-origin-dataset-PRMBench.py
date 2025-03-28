from datasets import load_dataset
import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_path", type=str, default="squad")
    parser.add_argument("--dataset_path", type=str, default="train")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    response_path = args.response_path
    dataset_path = args.dataset_path
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(os.path.dirname(response_path), f"{os.path.basename(response_path)}-{os.path.basename(dataset_path)}-merge.jsonl")
    
    print(f"Loading response dataset from {response_path}...")
    responses = load_dataset("json", data_files=response_path, split="train")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    origin_map = {}

    for data in dataset:
        if data["origin_id"] not in origin_map:
            origin_map[data["origin_id"]] = []
        origin_map[data["origin_id"]].append(data["id"])
    
    id_to_index = {}
    for i, data in enumerate(responses):
        id_to_index[data["id"]] = i
    
    new_dataset = []
    origin_keys = sorted(origin_map.keys())
    for origin_id in origin_keys:
        data_ids = origin_map[origin_id]
        item = {
            "id": origin_id,
            "yes_probability_average": [],
            "now_steps": []
        }
        for data_id in data_ids:
            index = id_to_index[data_id]
            item["yes_probability_average"].append(responses[index]["yes_probability_average"])
            item["now_steps"].append(responses[index]["now_step"])
        new_dataset.append(item)
    
    new_dataset = sorted(new_dataset, key=lambda x: x["id"])
    with open(output_path, "w") as f:
        for item in new_dataset:
            f.write(json.dumps(item) + "\n")