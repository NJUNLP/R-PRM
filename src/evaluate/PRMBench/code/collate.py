import json
import random
import argparse
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

dataset1, dataset2 = [], []

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, default="score.jsonl")
    parser.add_argument("--prm_dataset_path", type=str, default="prmbench_preview.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    path1 = args.score_path
    path2 = args.prm_dataset_path

    dataset1 = load_dataset(path1)
    dataset2 = load_dataset(path2)

    output_path = "../data/prm_result.jsonl"
    cnt = 0
    output_data = []
    for data1, data2 in zip(dataset1, dataset2):
        idx = data2['idx']
        scores = data1['yes_probability_average']
        labels = [score >= args.threshold for score in scores]
        classification = data2['classification']
        dic = {'scores': {'step_level_validity_scores': data1['yes_probability_average'], 'step_level_validity_labels': labels}, 'idx': f'{classification}_{idx}'}
        output_data.append(dic)

    with open(output_path, 'w') as f:
        for data in output_data:
            f.write(json.dumps(data) + "\n")