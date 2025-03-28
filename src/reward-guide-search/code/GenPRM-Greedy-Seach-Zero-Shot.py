import os
import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI

import regex

# 定义允许的字符集：
# - \p{Latin} 匹配所有拉丁字母
# - \p{Greek} 匹配所有希腊字母
# - \d 匹配数字
# - \s 匹配空白字符
# - 后面列出的是常用的数学符号、标点和 LaTeX 语法中常出现的符号
allowed_pattern = regex.compile(
    r"[^\p{Latin}\p{Greek}\d\s"
    r"\+\-\*/=\^\_\'\".,:;?!\(\)\{\}\[\]\\\$%<>|&@#"
    r"√∞±×÷°]"
)

def find_illegal_chars(text: str) -> list:
    """
    查找文本中不在允许范围内的字符，返回列表
    """
    return allowed_pattern.findall(text)

def is_math_answer_valid(answer: str) -> bool:
    """
    检查数学回答中是否存在非法字符：
      - 如果返回 True，则说明文本中没有不允许的字符
      - 如果返回 False，则文本中包含非法字符
    """
    illegal = find_illegal_chars(answer)
    if illegal:
        return False
    return True

def get_next_step(policy_model, question, previous_steps, policy_prompt, temperature):
    prompt = policy_prompt
    previous_step = "" if len(previous_steps) == 0 else "\n\n".join(previous_steps)
    now_prompt = prompt.format(question, previous_step)
    now_prompt = now_prompt.replace("left-brackets", "{").replace("right-brackets", "}")
    now_prompt += "\n\n"

    sampling_params = SamplingParams(
        n=8,
        top_p=0.95,
        temperature=temperature,
        max_tokens=512,
        stop=["<|eot_id|>", "\n\n"],
        include_stop_str_in_output=True,
    )
    llm_outputs = policy_model.generate(now_prompt, sampling_params)

    new_steps_candidates = []
    str_set = set()
    for output_obj in llm_outputs[0].outputs:
        gen_text = output_obj.text
        if not is_math_answer_valid(gen_text):
            continue
        if gen_text.endswith("\n\n"):
            gen_text = gen_text[:-2].strip()
        else:
            gen_text += "<|eot_id|>"
        gen_text =gen_text

        if gen_text in str_set:
            continue
        new_steps_candidates.append(gen_text)
        str_set.add(gen_text)
    # print(new_steps_candidates)
    # assert False
    return new_steps_candidates

def get_critique_prompt(question, previous_steps, now_step, critique_prompt):
    prompt = critique_prompt
    previous_step = (
        "\n".join(previous_steps)
        if len(previous_steps) > 0
        else "Since the Now Step is the first step, there are no Previous Steps."
    )
    return prompt.format(question, previous_step, now_step)


def get_reward(reward_model, tokenizer, question, previous_steps, now_step, critique, reward_prompt):
    prompt = reward_prompt
    previous_step = (
        "\n".join(previous_steps)
        if len(previous_steps) > 0
        else "Since the Now Step is the first step, there are no Previous Steps."
    )
    now_prompt = prompt.format(question, previous_step, now_step)

    if "Is the step correct (Yes/No)?" in critique:
        critique = critique.split("Is the step correct (Yes/No)?")[0] + "Is the step correct (Yes/No)?"
    else:
    
        critique = critique + "\nBased on your analysis. Is the step correct (Yes/No)?"
        print("Warning: critique does not contain 'Is the step correct (Yes/No)?'.")
        return -1

    now_prompt += critique

    input_ids = tokenizer.encode(now_prompt, return_tensors="pt").to(reward_model.device)
    with torch.no_grad():
        outputs = reward_model(input_ids)
        logits = outputs.logits

    last_token_logits = logits[0, -1, :]
    yes_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    relevant_logits = last_token_logits[[yes_token_id, no_token_id]]
    probabilities = torch.softmax(relevant_logits, dim=-1).tolist()

    return probabilities[0]


def main():
    parser = argparse.ArgumentParser(description="Iterative reasoning pipeline with reward model and critique model.")

    parser.add_argument("--policy_model_path", type=str, required=True, help="Path to the policy model.")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model.")


    parser.add_argument("--policy_prompt_path", type=str, required=True, help="Path to the policy prompt file.")
    parser.add_argument("--critique_prompt_path", type=str, required=True, help="Path to the critique prompt file.")
    parser.add_argument("--reward_prompt_path", type=str, required=True, help="Path to the reward prompt file.")


    parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset.")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--temperature", type=float, default=0.7, help="the temperature of the policy model.")
    parser.add_argument("--data_begin", type=int, default=0, help="Starting index of the dataset to process.")
    parser.add_argument("--data_end", type=int, default=None, help="Ending index of the dataset to process.")

    parser.add_argument("--port", type=int, default=8081, help="the port number.")
    parser.add_argument("--threshold", type=float, default=0.9, help="the threshold of the reward score.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.policy_prompt_path, "r", encoding="utf-8") as f:
        policy_prompt = f.read()
    with open(args.critique_prompt_path, "r", encoding="utf-8") as f:
        critique_prompt = f.read()
    with open(args.reward_prompt_path, "r", encoding="utf-8") as f:
        reward_prompt = f.read()

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = [dataset[i] for i in range(args.data_begin, args.data_end)]

    policy_model = LLM(
        model=args.policy_model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
    )

    reward_model = AutoModelForCausalLM.from_pretrained(
        args.reward_model_path, torch_dtype="bfloat16", device_map=f"cuda:1"
    )
    reward_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)

    print(f"http://localhost:{args.port}/v1")
    reward_critique = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")
    model = reward_critique.models.list().data[0].id


    new_dataset = []
    for i, data in tqdm(enumerate(dataset), desc="Processing dataset", total=len(dataset)):

        question = data["problem"]
        previous_steps = []
        used_steps = set()
        max_iteration = 30
        iteration_history = []
        iteration_index = 0

        while max_iteration > 0:
            max_iteration -= 1
            iteration_index += 1

            new_steps_candidates = get_next_step(policy_model, question, previous_steps, policy_prompt, args.temperature)
            #print(f"Question Number: {i}, Iteration: {iteration_index}, New Steps Candidates: {new_steps_candidates}")
            if len(new_steps_candidates) == 0:
                print(f"Question Number: {i}, No candidate generated at iteration {iteration_index}.")
                continue
            print(f"Question Number: {i}, Iteration: {iteration_index}, New Steps Candidates Number: {len(new_steps_candidates)}")
            new_step_accepted = False
            iteration_data = {
                "iteration_index": iteration_index,
                "candidates_info": [],
                "chosen_step_index": None,
                "chosen_step": None
            }

            for candidate_idx, candidate_step in enumerate(new_steps_candidates):
                if candidate_step in used_steps:
                    continue
                if candidate_step.endswith("<|eot_id|>"):
                    temp_candidate_step = candidate_step.replace("<|eot_id|>", "")
                else:
                    temp_candidate_step = candidate_step
                
                now_candidate_prompt = get_critique_prompt(question, previous_steps, temp_candidate_step, critique_prompt)

                chat_completion = reward_critique.chat.completions.create(
                    messages=[{"role": "user", "content": now_candidate_prompt}],
                    model=model,
                    temperature=0.95,
                    n=10,
                    max_tokens=3072,
                    top_p=0.95,
                )

                critique_texts = []
                rewards = []

                for choice in chat_completion.choices:
                    critique_text = choice.message.content
                    

                    reward_score = get_reward(
                        reward_model, tokenizer, question, previous_steps, temp_candidate_step, critique_text, reward_prompt
                    )
                    if reward_score == -1:
                        continue
                    rewards.append(reward_score)
                    critique_texts.append(critique_text)

                print(f"Question Number: {i}, Previous Steps Number: {len(previous_steps)}, Candidate Step Index: {candidate_idx}, Rewards: {rewards}")
                avg_reward = sum(rewards) / len(rewards)


                candidate_info = {
                    "candidate_step": candidate_step,
                    "critique_texts": critique_texts,
                    "rewards": rewards,
                    "avg_reward": avg_reward
                }
                iteration_data["candidates_info"].append(candidate_info)

                # if avg_reward > args.threshold:
                #     previous_steps.append(candidate_step)
                #     used_steps.add(candidate_step)
                #     new_step_accepted = True

                #     iteration_data["chosen_step_index"] = candidate_idx
                #     iteration_data["chosen_step"] = candidate_step
                #     break
                # else:
                #     used_steps.add(candidate_step)
            
            max_reward = -1
            max_reward_idx = -1
            for idx, candidate_info in enumerate(iteration_data["candidates_info"]):
                if candidate_info["avg_reward"] > max_reward:
                    max_reward = candidate_info["avg_reward"]
                    max_reward_idx = idx
            iteration_data["chosen_step_index"] = max_reward_idx
            iteration_data["chosen_step"] = iteration_data["candidates_info"][max_reward_idx]["candidate_step"]
            previous_steps.append(iteration_data["chosen_step"])
            iteration_history.append(iteration_data)

            #print(f"Question Number: {i}, Iteration: {iteration_index}, Chosen Step index: {iteration_data['chosen_step_index']}")

            if len(previous_steps) > 0 and "<|eot_id|>" in previous_steps[-1]:
                print(f"Question Number: {i}, Early stopping at iteration {iteration_index}.")
                break
            # if not new_step_accepted:
            #     if previous_steps:
            #         previous_steps.pop()
            #     print(f"Question Number: {i}, No new step accepted at iteration {iteration_index}. Retrying previous step.")

            # if len(previous_steps) > 0 and "<|eot_id|>" in previous_steps[-1]:
            #     print(f"Question Number: {i}, Early stopping at iteration {iteration_index}.")
            #     break

        new_dataset.append({
            "question": question,
            "iteration_history": iteration_history,
            "final_steps": previous_steps
        })

        output_file = os.path.join(args.output_dir, f"result-{args.data_begin}-{args.data_end}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    print(f"Done! Results are saved to {output_file}.")


if __name__ == "__main__":
    main()