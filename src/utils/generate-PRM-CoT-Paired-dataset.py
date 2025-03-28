import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

import numpy as np


def add_dataset_entry(target_file_path, dataset_name, file_name):
    """
    将新数据集条目添加到目标文件中。
    :param target_file_path: 目标 JSON 文件路径
    :param dataset_name: 数据集名称（作为键）
    :param file_name: 数据集文件路径
    """
    # 定义新条目结构
    new_entry = {
        dataset_name: {
            "file_name": file_name,
            "columns": {
            "prompt": "instruction",
            "response": "output"
            }
        }
    }

    # 读取现有文件内容
    try:
        with open(target_file_path, "r") as f:
            data = json.load(f)  # 读取文件内容为字典
    except FileNotFoundError:
        data = {}  # 如果文件不存在，初始化为空字典

    # 添加新条目
    if dataset_name in data:
        print(f"警告: 数据集 '{dataset_name}' 已存在，将覆盖其内容。")
    data.update(new_entry)

    # 将更新后的数据写回文件
    with open(target_file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"成功将 '{dataset_name}' 添加到 {target_file_path}")

def save_split(split, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in split:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(f"{json_line}\n")

def main(args):
    dataset = load_dataset("json", data_files=args.dataset_file, split="train")
    new_dataset = []
    instruction_set = set()

    correct_list = []
    error_list = []
    print("All dataset size: ",len(dataset))
    for data in dataset:
        output = data.get("chosen", "")
        has_yes = "Verification: Is the step correct (Yes/No)? Yes" in output
        has_no = "Verification: Is the step correct (Yes/No)? No" in output

        if has_yes:
            # print("Has yessssssssssssssssssss")
            # print(data.get("rejected", ""))
            # print("Verification: Is the step correct (Yes/No)? No" in data.get("rejected", ""))
            pass
            # if "Verification: Is the step correct (Yes/No)? No" not in data.get("rejected", ""):
            #     print(data.get("rejected", ""))
            # assert "Verification: Is the step correct (Yes/No)? No" in data.get("rejected", "")

        if has_no:
            # print("Has nooooooooooooooooo")
            # print(data.get("rejected", ""))
            # print("Verification: Is the step correct (Yes/No)? Yes" in data.get("rejected", ""))
            pass
            #assert "Verification: Is the step correct (Yes/No)? Yes" in data.get("rejected", "")

        if has_yes and has_no:
            continue

        new_dataset.append(data)

        if has_yes:
            correct_list.append(data)

        if has_no:
            error_list.append(data)

    min_length = min(len(correct_list),len(error_list))
    import random
    random.seed(42)
    random.shuffle(correct_list)
    random.shuffle(error_list)
    print("Original p/n ratio:", len(correct_list),len(error_list))
    correct_list = correct_list[:min_length]
    error_list = error_list[:min_length]
    print("Balanced p/n ratio:", len(correct_list),len(error_list))
    dataset = correct_list + error_list
    problem2data = {}
    from tqdm import tqdm
    for data in tqdm(dataset):
        problem = data["instruction"].split("\nPrevious Steps:")[0].split("\nQuestion: ")[1].strip()
        tem_list = problem2data.get(problem,[])
        tem_list.append(data)
        problem2data[problem] = tem_list

    problems = list(problem2data.keys())
    random.shuffle(problems)

    # Initialize training and test datasets
    test_dataset = []
    train_dataset = []

    # Add data from problems to test set until reaching desired size
    for problem in problems:
        if len(test_dataset) < args.test_size:
            test_dataset.extend(problem2data[problem])
        else:
            train_dataset.extend(problem2data[problem])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")


    save_split(train_dataset, args.train_filepath)
    save_split(test_dataset, args.test_filepath)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split dataset.")
    parser.add_argument('--dataset_file', type=str, required=True, help="Path to the input dataset JSONL file.")
    parser.add_argument('--train_filepath', type=str, required=True, help="Path to save the training dataset.")
    parser.add_argument('--test_filepath', type=str, required=True, help="Path to save the test dataset.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--test_size', type=int, default=2000, help="Number of samples per category in the test set.")
    parser.add_argument('--train_correct_size', type=int, default=66238, help="Number of correct samples in the training set.")
    parser.add_argument('--train_error_size', type=int, default=66238, help="Number of error samples in the training set.")
    parser.add_argument('--not_write_in_train', action='store_true', help="Do not write the training set to the data info.")
    parser.add_argument('--not_write_in_test', action='store_true', help="Do not write the test set to the data info.")

    args = parser.parse_args()
    main(args)
