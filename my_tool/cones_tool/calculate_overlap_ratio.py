import json
from collections import defaultdict


def calculate_overlap_ratio(dict1, dict2):
    # 获取两个字典的键名集合
    keys_dict1 = set(dict1.keys())
    keys_dict2 = set(dict2.keys())

    # 计算键名的交集
    intersection_keys = keys_dict1.intersection(keys_dict2)

    # 计算键名交集的比例
    test = keys_dict1.union(keys_dict2)
    overlap_ratio = len(intersection_keys) / len(keys_dict1.union(keys_dict2))

    return intersection_keys, overlap_ratio


json_path_1 = r'D:\seekoo\SD\sd-scripts\cones\e30-K30-th0-colorful_rhythm-concept_neurons.json'
json_path_2 = r'D:\seekoo\SD\sd-scripts\cones\e40-K30-th0-colorful_rhythm-general_neurons.json'
with open(json_path_1, 'r') as json_file:
    dict1 = json.load(json_file)
with open(json_path_2, 'r') as json_file:
    dict2 = json.load(json_file)

print(f"dict1:{len(dict1.keys())}")
print(f"dict2:{len(dict2.keys())}")
# # 示例数据
# dict1 = {'a': {'x': 1, 'y': 2}, 'b': {'z': 3}}
# dict2 = {'a': {'m': 5, 'n': 6}, 'c': {'p': 7}}

# 计算重合比例
intersection_keys, ratio = calculate_overlap_ratio(dict1, dict2)
print(f"键名重合的比例为: {ratio}")

save_dict = defaultdict(defaultdict)
