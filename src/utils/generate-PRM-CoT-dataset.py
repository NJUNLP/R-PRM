import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import random
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
    # new_dataset = []
    # instruction_set = set()
    # for data in dataset:
    #     if data["instruction"] in instruction_set:
    #         continue
    #     instruction_set.add(data["instruction"])
    #     new_dataset.append(data)
    # dataset = new_dataset

    count_correct, count_error = 0, 0
    new_dataset = []
    correct_list = []
    error_list = []
    ambiguous_count = 0

    for data in dataset:
        output = data.get("output", "")
        has_yes = "Verification: Is the step correct (Yes/No)? Yes" in output
        has_no = "Verification: Is the step correct (Yes/No)? No" in output

        if has_yes and has_no:
            ambiguous_count += 1
            continue
        if has_yes:
            target_str = "Verification: Is the step correct (Yes/No)? Yes"
        else:
            target_str = "Verification: Is the step correct (Yes/No)? No"

        def verify_string(text):
            # 定义两个可能的子串
            verification_yes = "Verification: Is the step correct (Yes/No)? Yes"
            verification_no = "Verification: Is the step correct (Yes/No)? No"

            yes_count = text.count(verification_yes)
            no_count = text.count(verification_no)
            
            if yes_count == 1 and no_count == 0:
                return True
            elif yes_count == 0 and no_count == 1:
                return True
            else:
                return False

        if verify_string(output)==False:
            ambiguous_count += 1
            continue

        
        occur_count = output.count(target_str)
        if occur_count != 1 or (output.rstrip(".").strip().endswith(target_str)==False):
            ambiguous_count += 1
            continue

                
        
        new_dataset.append(data)

        if has_yes:
            count_correct += 1
            correct_list.append(data)

        if has_no:
            count_error += 1
            error_list.append(data)

    print(f"Correct samples: {count_correct}")
    print(f"Error samples: {count_error}")
    print(f"Ambiguous samples excluded: {ambiguous_count}")
    print(f"Total processed samples: {len(new_dataset)}")

    min_length = min(len(correct_list),len(error_list))
    import random
    random.seed(42)
    random.shuffle(correct_list)
    random.shuffle(error_list)
    correct_list = correct_list[:min_length]
    error_list = error_list[:min_length]

    print(len(correct_list),len(error_list))
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

    # test_size_per_category = args.test_size
    # test_correct = correct_list[:test_size_per_category]
    # test_error = error_list[:test_size_per_category]
    # test_set = test_correct + test_error

    # train_correct_remaining = correct_list[test_size_per_category:]
    # train_error_remaining = error_list[test_size_per_category:]

    # train_set1_correct_size = args.train_correct_size
    # train_set1_error_size = args.train_error_size

    # train_set1_correct = train_correct_remaining[:train_set1_correct_size]
    # train_set1_error = train_error_remaining[:train_set1_error_size]
    # train_set1 = train_set1_correct + train_set1_error
    # print(len(train_set1))
    # if not args.not_write_in_train:
    #     save_split(train_set1, args.train_filepath)
    # if not args.not_write_in_test:
    #     save_split(test_set, args.test_filepath)

    # print("Datasets successfully created and saved.")
    # print(f"Test set samples: {len(test_set)}")

    # eval_instruction_set = set([data["instruction"] for data in test_set])
    # count = 0
    # for data in train_set1:
    #     if data["instruction"] in eval_instruction_set:
    #         count += 1
    # print(f"Test samples found in training set: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split dataset.")
    parser.add_argument('--dataset_file', type=str, required=True, help="Path to the input dataset JSONL file.")
    parser.add_argument('--train_filepath', type=str, required=True, help="Path to save the training dataset.")
    parser.add_argument('--test_filepath', type=str, required=True, help="Path to save the test dataset.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--test_size', type=int, default=2000, help="Number of samples per category in the test set.")


    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    main(args)
