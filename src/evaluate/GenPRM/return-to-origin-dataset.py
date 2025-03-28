import json
import argparse
from typing import List, Dict, Any
from datasets import load_dataset

def save_json(data: List[Dict[str, Any]], file_path: str):
    """
    Save data to a JSON file.

    :param data: List of dictionaries to save.
    :param file_path: Path to the output JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def merge_datasets(dataset1: List[Dict[str, Any]], dataset2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge dataset1 and dataset2 into the required structure.
    """
    idx_map = {}
    idx_origin_id = {}
    for i in range(len(dataset1)):
        idx_origin_id[dataset2[i]["data_id"]] = dataset2[i]["origin_id"]
        if dataset2[i]["data_id"] not in idx_map:
            idx_map[dataset2[i]["data_id"]] = {}
        if dataset2[i]["response_id"] not in idx_map[dataset2[i]["data_id"]]:
            idx_map[dataset2[i]["data_id"]][dataset2[i]["response_id"]] = []
        idx_map[dataset2[i]["data_id"]][dataset2[i]["response_id"]].append(i)

    new_dataset = []
    idx_map_list = sorted(idx_map.items(), key=lambda x: x[0])
    for data_id, response_map in idx_map_list:
        item = {
            "id": data_id,
            "origin_id": idx_origin_id[data_id],
            "yes_probabilities": [],
            "yes_probability_average": [],
            "yes_probability_average_min": [],
            "now_steps": [],
            "PRM-CoT": [],
            "yes_probability_average_last": []
        }
        response_map_list = sorted(response_map.items(), key=lambda x: x[0])
        for response_id, idx_list in response_map_list:
            yes_probabilities = []
            yes_probability_average = []
            now_steps = []
            PRM_CoT = []
            for idx in idx_list:
                yes_probabilities.append(dataset1[idx]["yes_probabilities"])
                yes_probability_average.append(dataset1[idx]["yes_probability_average"])
                now_steps.append(dataset1[idx]["now_step"])
                PRM_CoT.append(dataset1[idx]["PRM-CoT"])
            item["yes_probabilities"].append(yes_probabilities)
            item["yes_probability_average"].append(yes_probability_average)
            item["yes_probability_average_min"].append(min(yes_probability_average))
            item["yes_probability_average_last"].append(yes_probability_average[-1])
            # item["now_steps"].append(now_steps)
            # item["PRM-CoT"].append(PRM_CoT)
            
        new_dataset.append(item)
    return new_dataset

def main():
    parser = argparse.ArgumentParser(description="Merge two datasets into a new JSONL format.")
    parser.add_argument("--dataset1_path", type=str, required=True, help="Path to the first dataset JSONL file.")
    parser.add_argument("--dataset2_path", type=str, required=True, help="Path to the second dataset JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged dataset JSONL file.")
    args = parser.parse_args()

    # Load datasets
    print("Loading dataset1...")
    dataset1 = load_dataset("json", data_files=args.dataset1_path, split="train")
    print(f"Loaded {len(dataset1)} entries from dataset1.")

    print("Loading dataset2...")
    dataset2 = load_dataset("json", data_files=args.dataset2_path, split="train")
    print(f"Loaded {len(dataset2)} entries from dataset2.")

    # Merge datasets
    print("Merging datasets...")
    merged_dataset = merge_datasets(dataset1, dataset2)
    print(f"Merged dataset contains {len(merged_dataset)} entries.")

    # Save merged dataset
    print(f"Saving merged dataset to {args.output_path}...")
    save_json(merged_dataset, args.output_path)
    print("Merge complete.")

if __name__ == "__main__":
    main()
