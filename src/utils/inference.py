from datasets import load_dataset
from vllm import LLM, SamplingParams
import os
import argparse
from tqdm import tqdm
import json
from transformers import AutoTokenizer
import torch
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Inferencing settings")
    parser.add_argument("--temperature", default=0.2, type=float, help="the generation temperature")
    parser.add_argument("--gpu_memory_utilization", default=0.9, type=float, help="the gpu memory utilization")
    parser.add_argument("--n", default=1, type=int, help="the sample number of every question")
    parser.add_argument("--tensor_number", default=4, type=int, help="the tensor number")
    parser.add_argument("--data_begin", default=0, type=int, help="the number of data begin")
    parser.add_argument("--data_end", default=None, type=int, help="the number of data end")
    parser.add_argument("--dtype", default="float32", type=str, help="the tensor type")
    parser.add_argument("--divide_number", default=4, type=int, help="the divide number")
    parser.add_argument("--model_path", default=None, type=str, help="the model path")
    parser.add_argument("--prompt_path", default=None, type=str, help="the prompt path")
    parser.add_argument("--data_path", default=None, type=str, help="the data path")
    parser.add_argument("--write_path", default=None, type=str, help="the write path")
    parser.add_argument("--prompt_type", default="generation_answer", type=str, help="the prompt type")
    parser.add_argument("--max_token", default=8192, type=int, help="the max_token of SampleParams")
    parser.add_argument("--complement_key", default="response", type=str, help="the complement key value")
    parser.add_argument('--Sampling', action='store_true', help='Enable sampling mode if specified')
    return parser.parse_args()

def split_text(text):
    """清理文本，移除不必要的字符"""
    text = text.split("\n")
    text = [item.split("\\n") for item in text]
    text = [item.strip() for sublist in text for item in sublist]
    return "\n".join(text)

def load_json_prompt(prompt_path):
    """加载 JSON 格式的 prompt 文件"""
    with open(prompt_path, "r") as f:
        prompt = json.load(f)
    pre_messages = [{"role": "system", "content": prompt["system-prompt"]}] if 'system-prompt' in prompt else []
    for example in prompt["examples"]:
        pre_messages.append({"role": "user", "content": example["question"]})
        pre_messages.append({"role": "assistant", "content": example["answer"]})
    return prompt.get("template", ""), pre_messages

def build_prompt(data, prompt_type, template, complement_key):
    """根据 prompt_type 和模板生成 prompt"""
    prompt_map = {
        "generation_answer": [data.get("query", "")],
        "GenPRM-CoT-Generate": [
            data.get("question", ""),
            data.get("previous_steps", ""),
            data.get("now_step", "")
        ],
        "GenPRM-CoT-Generate-Origin": [
            data.get("query", ""),
            data.get("previous-steps", ""),
            data.get("now-step", "")
        ],
        "critic-processbench": [data.get("problem", ""), data.get("steps_merge", "")],
        "PRM-CoT": [
            data.get("query", ""),
            data.get("previous-steps", ""),
            data.get("now-step", ""),
            data.get("expert-judge", "")
        ]

    }
    # 使用安全的 get 方法避免字段缺失引发 KeyError
    # print(prompt_type)
    # print(data)
    prompt_args = prompt_map.get(prompt_type, [])
    # print(prompt_args)
    # assert False
    return template.format(*prompt_args)


if __name__ == "__main__":
    # print(sys.argv)
    args = parse_args()
    temperature = args.temperature
    top_p = 0.95
    sample = SamplingParams(
        n=args.n,
        top_p=top_p,
        temperature=temperature,
        max_tokens=args.max_token,
        stop=["<|eot_id|>"],
        include_stop_str_in_output=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_path = args.model_path
    prompt_path = args.prompt_path
    data_path = args.data_path
    write_path = args.write_path

    if not os.path.exists(write_path):
        os.makedirs(write_path, exist_ok=True)
    
    cmd = " ".join(sys.argv)
    with open(os.path.join(write_path, "cmd.txt"), "w") as f:
        f.write(cmd)

    template, pre_messages = load_json_prompt(prompt_path) if prompt_path.endswith(".json") else (open(prompt_path).read().strip(), [])

    dataset = []
    # if data_path.endswith(".jsonl"):
    #     with open(data_path, "r") as f:
    #         dataset = [json.loads(line) for line in f]
    # else:
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    response_name = None
    end = args.data_end if args.data_end is not None else len(dataset)


    model_name = model_path.split("/")[-1]
    

    if args.Sampling:
        response_name = f"{model_name}-response-{args.data_path.split('/')[-1].split('.')[0]}.jsonl" 
    else:
        response_name = f"{model_name}-response-{args.data_begin}-{end}.jsonl"
    
    print("======model name========",response_name)
    dataset = [dataset[i] for i in range(args.data_begin, end)]
    prompts = []
    for data in dataset:
        now_prompt = build_prompt(data, args.prompt_type, template, args.complement_key)
        if prompt_path.endswith(".json"):
            now_prompt = now_prompt.replace("left-brackets", "{").replace("right-brackets", "}")
            messages = pre_messages + [{"role": "user", "content": now_prompt}]
            final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(final_prompt)
        else:
            now_prompt = now_prompt.replace("left-brackets", "{").replace("right-brackets", "}")
            prompts.append(now_prompt)

    print(prompts[0])
    print("=============")
    print(prompts[1])
    print("=============")
    print(prompts[-1])
    # exit()
    
    model = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_number,
        dtype=args.dtype,
        swap_space=0,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        # max_model_len=40000
    )
    response = model.generate(prompts, sample)
    

    with open(os.path.join(write_path, response_name), "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            item = {"id": dataset[i]["id"], "response": []}
            try: 
                for j in range(args.n):
                    content = split_text(response[i].outputs[j].text.strip().strip("\\").strip("\\n").strip())
                    item["response"].append(content)
            except:
                item["response"] = ["Verification: Is the step correct (Yes/No)?" for _ in range(args.n)]
            finally:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
