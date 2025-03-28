import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def main(args):
    dataset = load_dataset("json", data_files=args.dataset_file, split="train")
    responses = load_dataset("json", data_files=args.responses_file, split="train")
    
    idx_map = {}
    for i in range(len(dataset)):
        idx_map[dataset[i]['id']] = i

    correct_count = 0
    other_count = 0
    wrong_count_amount = 0
    wrong_count_correct = 0
    wrong_count_error = 0
    error_dataset = []

    datas_we_skip = 0
    new_dataset = [data for data in dataset]

    for i in tqdm(range(len(responses))):
        idx = idx_map[responses[i]['id']]
        PRM_CoT = None
        target_list = []
        if dataset[idx]["expert-judge"] in ["Positive", "Negative"]:

            target_str = "Verification: Is the step correct (Yes/No)? Yes"
            if dataset[idx]["expert-judge"] == "Negative":
                target_str = "Verification: Is the step correct (Yes/No)? No"

            now_count = 0
            for response in responses[i]['response']:
                if "Verification: Is the step correct (Yes/No)? Yes" in response and "Verification: Is the step correct (Yes/No)? No" in response:
                    continue

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

                if verify_string(response)==False:
                    datas_we_skip += 1
                    continue

                if target_str in response:
                    occur_count = response.count(target_str)
                    if occur_count == 1 and response.rstrip(".").strip().endswith(target_str):
                        now_count += 1
                        target_list.append(response)
    

            if now_count > 0:
                correct_count += 1
            else:
                wrong_count_amount += 1
                error_dataset.append(dataset[idx])
                if dataset[idx]["expert-judge"] == "Positive":
                    wrong_count_correct += 1
                elif dataset[idx]["expert-judge"] == "Negative":
                    wrong_count_error += 1
        else:
            raise ValueError("Unexpected value in 'expert-judge': {}".format(dataset[idx]["expert-judge"]))

        if len(target_list) > 0:
            #target_list = sorted(target_list, key=lambda x: len(tokenizer(x)['input_ids']))
            PRM_CoT = target_list[-1]

        if "PRM-CoT" not in new_dataset[idx]:
            new_dataset[idx]["PRM-CoT"] = None

        assert new_dataset[idx]["PRM-CoT"] is None
        new_dataset[idx]["PRM-CoT"] = PRM_CoT

    with open(args.prompt_file, "r") as f:
        prompt = f.read()

    final_dataset = []
    for data in new_dataset:
        if "PRM-CoT" not in data or data["PRM-CoT"] is None:
            continue
        has_yes = "Verification: Is the step correct (Yes/No)? Yes" in data["PRM-CoT"]
        has_no = "Verification: Is the step correct (Yes/No)? No" in data["PRM-CoT"]
        if has_yes and has_no:
            continue
        if data["PRM-CoT"] is None:
            continue
        final_dataset.append({
            "instruction": prompt.format(data["query"], data["previous-steps"], data["now-step"]),
            "output": data["PRM-CoT"]
        })

    print(f"Correct Count: {correct_count}")
    print(f"Other Count: {other_count}")
    print(f"Wrong Count Amount: {wrong_count_amount}")
    print(f"Wrong Count Correct: {wrong_count_correct}")
    print(f"Wrong Count Error: {wrong_count_error}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Ok number: {correct_count + other_count}")
    print(f"Length Error Dataset: {len(error_dataset)}")
    print(f"datas_we_skip : {datas_we_skip}")
    count = 0
    for data in final_dataset:
        if data["output"] is not None:
            count += 1
    print(f"Count: {count}")
    length = len(final_dataset) // 1000
    reponse_file_name = args.responses_file.split("/")[-1].replace(".jsonl","")
    output_file = os.path.join(args.output_dir, f"{reponse_file_name}-{length}k.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in final_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Output file: {output_file}")

    with open(output_file.replace(".jsonl","ErrorCase.jsonl"), 'w', encoding='utf-8') as f:
        for data in error_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    instruction_set = set()
    # decomination_dataset = []
    correct_number = 0
    error_number = 0
    for data in final_dataset:
        if data["instruction"] in instruction_set:
            continue
        instruction_set.add(data["instruction"])
        if "Verification: Is the step correct (Yes/No)? No" in data["output"]:
            error_number += 1
        elif "Verification: Is the step correct (Yes/No)? Yes" in data["output"]:
            correct_number += 1
    print(f"去重后数据集中正确样例的个数：{correct_number}")
    print(f"去重后数据集中错误样例的个数：{error_number}")
    print(f"去重后数据集中样例的个数：{len(instruction_set)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and responses.")
    parser.add_argument('--dataset_file', type=str, required=True, help="Path to the input dataset JSONL file.")
    parser.add_argument('--responses_file', type=str, required=True, help="Path to the input responses JSONL file.")
    parser.add_argument('--prompt_file', type=str, required=True, help="Path to the prompt text file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dir to save the processed dataset.")

    args = parser.parse_args()
    main(args)