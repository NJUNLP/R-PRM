import os
import json
import numpy as np
import argparse
from datasets import load_dataset

os.chdir(os.path.dirname(__file__))
print("当前代码运行所在的路径为：", os.getcwd())

def compute_match(responses, threshold=0.5):
    """
    计算responses中第一个小于0.2的索引，如果不存在则返回-1。
    """
    for idx, value in enumerate(responses):
        if value < threshold:
            return idx
    return -1

def process_dataset(input_dataset, config, output_dir='output', threshold=0.5):
    """
    处理数据集，计算match值，分类数据，并计算acc1、acc2和f1分数。

    参数：
    - input_dataset: datasets.Dataset对象
    - config: 配置名称，用于输出文件命名
    - output_dir: 输出文件夹名称（默认是'output'）
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    res_data = []
    
    # 遍历数据集并计算'match'值
    for data in input_dataset:
        # 计算'match'值
        match = compute_match(data.get('responses', []), threshold)
        # 将'match'添加到数据中
        data_dict = dict(data)
        match = 1 if match == data.get('label', -1) else 0
        data_dict['match'] = match
        res_data.append(data_dict)
    
    # 根据'label'分类数据
    error_data = [e for e in res_data if e.get('label', -1) != -1]
    correct_data = [e for e in res_data if e.get('label', -1) == -1]
    
    # 将error_data写入JSONL文件
    error_file = os.path.join(output_dir, f'error.jsonl')
    print(f'error_file: {error_file}')
    # assert False
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, ensure_ascii=False, indent=4)
        # for e in error_data:
        #     f.write(json.dumps(e, ensure_ascii=False) + '\n')
    
    # 将correct_data写入JSONL文件
    correct_file = os.path.join(output_dir, f'correct.jsonl')
    print(f'correct_file: {correct_file}')
    with open(correct_file, 'w', encoding='utf-8') as f:
        json.dump(correct_data, f, ensure_ascii=False, indent=4)
        # for e in correct_data:
        #     f.write(json.dumps(e, ensure_ascii=False) + '\n')
    
    # 计算acc1和acc2
    acc1 = np.mean([e['match'] for e in error_data]) * 100 if error_data else 0
    acc2 = np.mean([e['match'] for e in correct_data]) * 100 if correct_data else 0
    
    # 计算f1分数
    if acc1 + acc2 == 0:
        f1 = 0
    else:
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    
    # 打印结果
    print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')
    
    # 将结果保存到JSON文件
    result = {
        'config': config,
        'error_acc': acc1,
        'correct_acc': acc2,
        'f1': f1
    }
    
    result_file = os.path.join(output_dir, f'result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="计算分数并分类数据集")
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help="输入数据集的路径（支持JSONL格式）"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help="输出文件夹名称（默认是'output'）"
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help="阈值"
    )
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_dir = args.output_dir
    if output_dir == 'output':
        output_dir = os.path.join(os.path.dirname(input_path), output_dir)
        print(f'output_dir: {output_dir}')
        # assert False
    
    # 使用load_dataset加载JSONL数据集
    dataset = load_dataset('json', data_files=input_path, split='train')
    
    # 处理数据集
    process_dataset(dataset, None, output_dir=output_dir, threshold=args.threshold)

if __name__ == '__main__':
    main()
