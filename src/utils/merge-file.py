# 这个文件需要你给定一个文件夹，然后将改文件夹下的jsonl文件合并，然后输出该文件夹下，要求文件都有id字段
# 使用方法 python merge-file.py <dir_path> 

import os
import json
import sys
os.chdir(os.path.dirname(__file__))
print("当前代码运行所在的路径为：", os.getcwd())
dir_path = sys.argv[1]

if os.path.exists(os.path.join(dir_path, 'response.jsonl')):
    os.remove(os.path.join(dir_path, 'response.jsonl'))
file_names = os.listdir(dir_path)
file_names = [f for f in file_names if f.endswith('.jsonl')]

merge_file = []
for filename in file_names:
    with open(os.path.join(dir_path, filename), 'r') as f:
        for line in f:
            merge_file.append(json.loads(line))
merge_file = sorted(merge_file, key=lambda x: x['id'])
with open(os.path.join(dir_path, 'response.jsonl'), 'w') as f:
    for line in merge_file:
        f.write(json.dumps(line) + '\n')