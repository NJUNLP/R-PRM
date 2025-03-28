import json
import argparse
import os
from datasets import load_dataset
from collections import defaultdict
os.chdir(os.path.dirname(__file__))
print("当前代码运行所在的路径为：", os.getcwd())

def parse_arguments():
    parser = argparse.ArgumentParser(description="处理数据集并生成最终输出文件。")
    
    parser.add_argument(
        '--data_name',
        type=str,
        required=True,
        help="数据集名称，例如 'olympiadbench'。"
    )
    
    parser.add_argument(
        '--response_path',
        type=str,
        required=True,
        help="响应文件的路径，例如 '/path/to/response.jsonl'。"
    )
    
    parser.add_argument(
        '--final_output_file',
        type=str,
        default=None,
        help="最终输出文件的路径。默认与 response_path 同目录，文件名为 'final_result.json'。"
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    data_name = args.data_name
    response_file = args.response_path
    
    # 根据 data_name 动态确定 original_file 和 new_dataset_file 的路径
    original_file = f"../../datasets/ProcessBench/{data_name}.json"
    new_dataset_file = f"../../datasets/ProcessBench/{data_name}_filtered.json"
    
    # 确定 final_output_file
    if args.final_output_file:
        final_output_file = args.final_output_file
    else:
        response_dir = os.path.dirname(response_file)
        final_output_file = os.path.join(response_dir, "final_result.json")
    
    # 检查文件是否存在
    for file_path in [original_file, new_dataset_file, response_file]:
        if not os.path.exists(os.path.abspath(file_path)):
            print(f"错误: 文件 {file_path} 不存在。请检查路径是否正确。")
            return
    
    # 加载原始数据集
    original_dataset = load_dataset(
        "json",
        data_files=original_file,
        split="train",
    )
    
    # 加载中间数据集
    new_dataset = load_dataset(
        "json",
        data_files=new_dataset_file,
        split="train",
    )
    
    # 加载响应数据集
    response_dataset = load_dataset(
        "json",
        data_files=response_file,
        split="train",
    )
    
    # 创建 'id' 到 'response' 的映射
    id_to_response = {item['id']: (item["id"], item['yes_probability_average']) for item in response_dataset}
    id_to_PRM_CoT = {item['id']: (item["id"], item["PRM-CoT"][0]) for item in response_dataset}
    # 创建 'origin_id' 到所有相关 'response' 的映射
    origin_id_to_responses = defaultdict(list)
    origin_id_to_PRM_CoT = defaultdict(list)
    
    for item in new_dataset:
        origin_id = item['origin_id']  # 对应原始数据集的 'id'（字符串）
        step_id = item['id']           # 对应 response.json 的 'id'（数字）
        response = id_to_response.get(step_id, "")
        PRM_CoT = id_to_PRM_CoT.get(step_id, "")
        origin_id_to_responses[origin_id].append(response)
        origin_id_to_PRM_CoT[origin_id].append(PRM_CoT)
    
    # 检查每个 'origin_id' 的步骤数量是否匹配
    # 并合并数据
    final_dataset = []
    
    for original_item in original_dataset:
        origin_id = original_item['id']  # 原始数据集中的 'id'（字符串）
        responses = origin_id_to_responses.get(origin_id, [])
    
        # 获取步骤数量
        steps = original_item.get('steps', [])
        # if len(responses) != len(steps):
        #     print(f"警告: 'origin_id' 为 {origin_id} 的响应数量 ({len(responses)}) 与步骤数量 ({len(steps)}) 不匹配。")
    
        # 创建新的数据条目
        responses = sorted(responses, key=lambda x: x[0])
        responses = [response[1] for response in responses]

        # 获取 PRM_CoT
        PRM_CoT = origin_id_to_PRM_CoT.get(origin_id, [])
        PRM_CoT = sorted(PRM_CoT, key=lambda x: x[0])
        PRM_CoT = [cot[1] for cot in PRM_CoT]

    
        new_item = {
            **original_item,  # 包含原始数据集的所有字段
            "responses": responses,  # 新增 'responses' 字段
            "PRM_CoT": PRM_CoT  # 新增 'PRM_CoT' 字段
        }
    
        final_dataset.append(new_item)
    
    # 将最终数据集保存为 JSON 文件
    try:
        with open(final_output_file, "w", encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        print(f"新的数据集已保存至 {final_output_file}")
    except Exception as e:
        print(f"错误: 无法保存最终数据集至 {final_output_file}。详细信息: {e}")

if __name__ == "__main__":
    main()
