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
    full_count = 0
    other_count = 0
    wrong_count_amount = 0
    wrong_count_correct = 0
    wrong_count_error = 0
    error_dataset = []
    new_dataset = [data for data in dataset]
    datas_we_skip = 0
    positive_sample = 0
    negtive_sample = 0
    chosen_count = 0

    for i in tqdm(range(len(responses))):
        idx = idx_map[responses[i]['id']]
        PRM_CoT = None
        target_list = []
        if dataset[idx]["expert-judge"] in ["Positive", "Negative"]:

            target_str = "Verification: Is the step correct (Yes/No)? Yes"
            if dataset[idx]["expert-judge"] == "Negative":
                target_str = "Verification: Is the step correct (Yes/No)? No"

            now_count = 0
            correct_response_list = []
            error_response_list = []
            valid = 0
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
                    
                valid += 1
                if target_str in response:
                    occur_count = response.count(target_str)
                    if occur_count == 1 and response.rstrip(".").strip().endswith(target_str):
                        now_count += 1
                        target_list.append(response)
                        correct_response_list.append(response)
                        chosen_count += len(correct_response_list)
                    else:
                        continue
                else:
                    error_response_list.append(response)

            if now_count > 0:
                correct_count += 1
                if len(error_response_list) ==0:
                    full_count += 1
            else:
                wrong_count_amount += 1
                error_dataset.append(dataset[idx])
                if dataset[idx]["expert-judge"] == "Positive":
                    wrong_count_correct += 1
                elif dataset[idx]["expert-judge"] == "Negative":
                    wrong_count_error += 1
        else:
            raise ValueError("Unexpected value in 'expert-judge': {}".format(dataset[idx]["expert-judge"]))
        PRM_CoT_pair = []

        if len(target_list) > 0 and len(error_response_list) > 0:
            for correct_response_item in correct_response_list:
                for error_response_item in error_response_list:
                    PRM_CoT_pair.append([correct_response_item, error_response_item])

            if args.filter:
                if valid < (len(target_list) * 3):
                    PRM_CoT_pair = PRM_CoT_pair[:2]
                else:
                    PRM_CoT_pair = None

        if "PRM-CoT-pair" not in new_dataset[idx]:
            new_dataset[idx]["PRM-CoT-pair"] = None


        assert new_dataset[idx]["PRM-CoT-pair"] is None
        new_dataset[idx]["PRM-CoT-pair"] = PRM_CoT_pair

    with open(args.prompt_file, "r") as f:
        prompt = f.read()

    final_dataset = []
    for data in new_dataset:
        if "PRM-CoT-pair" not in data or data["PRM-CoT-pair"] is None:
            continue
        if len(data["PRM-CoT-pair"]) == 0:
            continue

        for item in data["PRM-CoT-pair"]:
            chosen = item[0]
            rejected = item[1]
            if "Verification: Is the step correct (Yes/No)? Yes" in chosen and "Verification: Is the step correct (Yes/No)? No" in chosen:
                continue
            if "Verification: Is the step correct (Yes/No)? Yes" in rejected and "Verification: Is the step correct (Yes/No)? No" in rejected:
                continue 
                
            if "Verification: Is the step correct (Yes/No)? Yes" in chosen:
                positive_sample += 1
            else:
                negtive_sample += 1
            final_dataset.append(
                {
                    "instruction": prompt.format(data["query"], data["previous-steps"], data["now-step"]),
                    "chosen": chosen,
                    "rejected": rejected
                }
            )



    print(f"chosen_count: {chosen_count}")
    print(f"Correct Count: {correct_count}")
    print(f"Full Count: {full_count}")
    print(f"positive_sample : {positive_sample}")
    print(f"negtive_sample : {negtive_sample}")
    print(f"Other Count: {other_count}")
    print(f"Wrong Count Amount: {wrong_count_amount}")
    print(f"Wrong Count Correct: {wrong_count_correct}")
    print(f"Wrong Count Error: {wrong_count_error}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Ok number: {correct_count + other_count}")
    print(f"Length Error Dataset: {len(error_dataset)}")
    print(f"datas_we_skip : {datas_we_skip}")
    # with open(args.error_file, 'w', encoding='utf-8') as f:
    #     for data in error_dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + '\n')


    length = len(final_dataset) // 1000

    reponse_file_name = args.responses_file.split("/")[-1].replace(".jsonl","")
    output_file = os.path.join(args.output_dir, f"{reponse_file_name}-{length}k-pair.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in final_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Output file: {output_file}")
    print(f"收集到 {len(final_dataset)} 对数据，并写入 {os.path.join(args.output_dir, 'GenPRM-{}k-pair.jsonl'.format(length))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and responses.")
    parser.add_argument('--dataset_file', type=str, required=True, help="Path to the input dataset JSONL file.")
    parser.add_argument('--responses_file', type=str, required=True, help="Path to the input responses JSONL file.")
    parser.add_argument('--prompt_file', type=str, required=True, help="Path to the prompt text file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dir to save the processed dataset.")
    parser.add_argument("--error_file", type=str, default="error.jsonl", help="Path to the error file.")
    parser.add_argument('--filter', action='store_true', help='Enable sampling mode if specified')
    args = parser.parse_args()
    main(args)