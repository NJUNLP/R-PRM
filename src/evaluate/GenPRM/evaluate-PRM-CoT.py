from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
from datetime import datetime
import argparse
import os
import json

os.chdir(os.path.dirname(__file__))
print("当前代码运行所在的路径为：", os.getcwd())

def parse_args():
    parser = argparse.ArgumentParser(description="PRM Reward Calculation")
    parser.add_argument("--begin", default=None, type=int, help="piece begin number")
    parser.add_argument("--end", default=None, type=int, help="piece end number")
    parser.add_argument("--type", default="bfloat16", type=str, help="torch dtype")
    parser.add_argument("--data_name", default="custom_data", type=str, help="the name of data")
    parser.add_argument("--model_path", required=True, type=str, help="path to the model")
    parser.add_argument("--data_path", required=True, type=str, help="path to the data")
    parser.add_argument("--response_path", required=True, type=str, help="path to the responses")
    parser.add_argument("--output_path", required=True, type=str, help="path to save outputs")
    parser.add_argument("--temperature", default=0.0, type=float, help="temperature for sampling")
    parser.add_argument("--prompt_path", required=True, type=str, help="path to prompt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    begin, end = args.begin, args.end
    model_name = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("model_path:", os.path.abspath(model_name))
    print("data_path:", os.path.abspath(args.data_path))
    print("response_path:", os.path.abspath(args.response_path))
    print("output_path:", os.path.abspath(args.output_path))

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=args.type, device_map="auto")
    
    now_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    model_name_split = os.path.basename(model_name)
    log_path = os.path.join("../../logs/PRM-Rewards", args.data_name, model_name_split)
    os.makedirs(log_path, exist_ok=True)
    logger.add(os.path.join(log_path, f"{begin}-{end}-{now_time}.log"), rotation="100 MB", retention="7 days", compression="zip", level="DEBUG")

    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data = load_dataset("json", data_files=args.data_path, split="train")
    responses = load_dataset("json", data_files=args.response_path, split="train")
    
    data = [data[i] for i in range(begin, end)]
    # responses = [responses[i] for i in range(begin, end)]
    idx_map = {}
    for i, response in enumerate(responses):
        idx_map[response["id"]] = i
    responses = [responses[idx_map[data[i]["id"]]] for i in range(len(data))]

    logger.info(f"Loaded {len(data)} data items and {len(responses)} responses.")
    with open(args.prompt_path, "r") as f:
        prompt = f.read()

    save_dataset = []
    model.eval()

    for i in tqdm(range(len(data))):
        question = data[i]["question"]
        item = {
            "id": data[i]["id"],
            "query": question,
            "previous_steps": data[i]["previous_steps"],
            "now_step": data[i]["now_step"],
            "PRM-CoT": responses[i]["response"],
            "yes_probabilities": [],
            "yes_probability_average": 0.0
        }
        
        yes_probability_sum = 0.0
        for j, response in enumerate(responses[i]["response"]):
            if "Is the step correct (Yes/No)?" in response:
                response = response.split("Is the step correct (Yes/No)?")[0] + "Is the step correct (Yes/No)?"
            else:
                response = response + "\nBased on your analysis. Is the answer correct (Yes/No)?"
            formatted_prompt = prompt.format(question, data[i]["previous_steps"], data[i]["now_step"]) + response
            
            input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            last_token_logits = logits[0, -1, :]
            yes_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
            no_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]
            relevant_logits = last_token_logits[[yes_token_id, no_token_id]]
            probabilities = torch.softmax(relevant_logits, dim=-1).tolist()
            
            yes_probability = probabilities[0]
            item["yes_probabilities"].append(yes_probability)
            yes_probability_sum += yes_probability
            #logger.info(f"Item {i}, Response: {j}... - Yes Probability: {yes_probability:.4f}")

        item["yes_probability_average"] = yes_probability_sum / len(responses[i]["response"])
        save_dataset.append(item)

    output_dir = os.path.join(args.output_path, f"{model_name_split}-{args.data_name.upper()}-PRM-Rewards-{args.temperature}")
    os.makedirs(output_dir, exist_ok=True)
    save_file = os.path.join(output_dir, f"{begin}-{end}-{now_time}.jsonl")

    with open(save_file, "w") as f:
        for record in save_dataset:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Processing complete. Results saved to {save_file}")
