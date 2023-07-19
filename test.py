# import torch.nn as nn
# import torch
#
#
# def watch_weights(weights_dict):
#     for name, weight in weights_dict.items():
#         print(name, weight)
#
#
# linear = nn.Linear(3, 3)
# print(linear.weight)
# weights_test = linear.state_dict()
# print('org weights:')
# watch_weights(weights_test)
#
# a, b = torch.randn_like(linear.weight), torch.randn_like(linear.bias)
# print(f"randn create:{a}\n{b}")
# weights_dict = {'weight': a, 'bias': b}
#
# # linear.load_state_dict(weights_dict)
# # print(f'load_state_dict:')
# # watch_weights(linear.state_dict())
#
# linear.weight.data.copy_(weights_dict['weight'])
# print("after:")
# watch_weights(linear.state_dict())
#
# a = [1, 2]
# [b, c] = a
# print(b, c)

# from safetensors.torch import load_file
# path = r'D:\seekoo\SD\sd-scripts\result\renwu_full.safetensors'
# lora_sd = load_file(path)
# # print(lora_sd)
# unet_dict = {}
# te_dict = {}
# for key, value in lora_sd.items():
#     if 'lora_unet' in key and 'lora_down' in key:
#         unet_dict[key] = value
#     if 'lora_te'in key and 'lora_down' in key:
#         te_dict[key] = value
# print(unet_dict)

# test_list = [0, 0.5, 1]
# groups = []
# for i in test_list:
#     for j in test_list:
#         groups.append([i, j])
#
# for group in groups:
#     test = " ".join(map(str, group))
#     print(test)
    # print(f"{*group}")

# test = ['1', '0.2']
# res = list(map(float, test))
# print(res)

import itertools
# import numpy as np
# #
# A = np.arange(0, 1.5, 0.25).tolist()
# print(A)
# print(A[1:-1])
# combinations = list(itertools.product(A, repeat=2))
# #
# for combination in combinations:
#     print(combination)
# print(len(combinations))

# a = r'./asd/asd.12'
# print(a.endswith('.12'))


# import random
# import itertools
# import numpy as np
# a = np.arange(0, 1.5, 0.5).tolist()
# combinations = list(itertools.product(a, repeat=2))
# print(combinations)
# sample_list = random.sample(combinations, k=2)
# print(sample_list)

# merge_num = 3
# print([1/float(merge_num)] * int(merge_num))

# n = 4
# output = [[0] * n for _ in range(n)]
# for i in range(n):
#     output[i][n - i - 1] = 1
#
# print(output)


# import random
#
# def sample_without_replacement(elements, n):
#     if n > len(elements):
#         raise ValueError("n must be less than or equal to the number of elements")
#
#     # 使用 random.sample 函数来从 elements 列表中随机选择 n 个不重复的样本
#     samples = random.sample(elements, n)
#     return samples
#
# # 测试代码
# elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# n = 10
# samples = sample_without_replacement(elements, n)
# print(samples)


# import random
#
#
# def generate_lists(m, n):
#     result = []
#     for _ in range(n):
#         random_list = [random.random() for _ in range(m)]  # 生成m个介于0到1之间的随机数
#         total = sum(random_list)
#         normalized_list = [num / total for num in random_list]  # 归一化处理，使列表元素相加为1
#         result.append(normalized_list)
#     return result
# m = 5  # 列表中的元素个数
# n = 3  # 列表的个数
# lists = generate_lists(m, n)
# for lst in lists:
#     print(lst)


# import cv2
# # path = r'E:\Data\concat_pic\backup\7.9-10_lora_merge\MultiMerge_10_heali_0.1_oldsc_0.1_beaut_0.1_renwu_0.1_color_0.1_vecto_0.1_flat__0.1_fight_0.1_color_0.1_xinha_0.1_\A black athlete is preparing to dunk as he leaps through the air with a basketball gym in the background_None.png'
# path = r'1child with hat, yellow cat, street, sunny day_None.png'
# name = path.replace(', ', '_').replace(' ', '_')
# print(name)
# # img = cv2.imread(path)
# # cv2.imshow('Image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


import os
# # import cv2
# import shutil
# from tqdm import tqdm
# import copy
# path = r'E:\Data\concat_pic\backup\7.9-10_lora_merge'
# img_dirs = [os.path.join(path, img_dir) for img_dir in os.listdir(path)]
#
# for img_dir in tqdm(img_dirs):
#     for img_name in os.listdir(img_dir):
#         img_path = os.path.join(img_dir, img_name)
#         print(os.path.isfile(img_path))
#         if ', ' in img_name:
#             save_name = img_name.replace(', ', '-').replace(' ', '_')
#             save_path = os.path.join(img_dir, save_name)
#
#             # img = cv2.imread(img_path)
#             # cv2.imwrite(save_path, img)
#
#             # shutil.copy2(img_path, save_path)
#
#             os.rename(img_path, save_path)

# dir = r'E:\Data\concat_pic\backup\7.9-10_lora_merge\MultiMerge_10_heali_0.1_oldsc_0.1_beaut_0.1_renwu_0.1_color_0.1_vecto_0.1_flat__0.1_fight_0.1_color_0.1_xinha_0.1_'
# print(os.path.isdir(dir))
# name = r'A man is skiing with snow in the background_None.png'
# print(os.path.isfile(os.path.join(dir, name)))

#
import numpy as np
# a = np.arange(0, 8, 1).reshape(2, 4).tolist()
# print(a)
# a = [[1, 0], [1, 0], [0.25, 0.75]]
# print(a)
# print(np.array(a).reshape(-1).tolist())

# a = 'sd\\sd\\'
# print(a.replace('\\', ' '))

# import os
# path = r'E:\\Data\\test\\OverFitting\\beautiful_as_crystal_wo_te'
# if not os.path.exists(path):
#     os.mkdir(path)


# a = 8.15432168
# print('{:2.4f}'.format(a))

# import torch
# data = torch.tensor([4.9276e-05, -4.7397e-05, -4.9604e-05, -4.9115e-05])
# grad = torch.tensor([4.4757e-07,  7.2846e-07,  7.9386e-07, -7.9607e-07])
# # torch.ones_like(data)
# mul = torch.mul(0.5 * data, grad)
# mul_64float = mul.clone().detach().to(torch.float64)
# print(mul_64float)
#
# result = 1 - mul_64float
# print(result)
# print('{:.32f}'.format(result[0]))

# import torch.nn as nn
# a = nn.Conv2d(2, 3, kernel_size=3, stride=1)
# b = nn.Linear(4, 3)
# print(a)

import torch
data = torch.tensor([[4.9276e-05, -4.7397e-05], [-4.9604e-05, -4.9115e-05]])
# data = data.tolist()
print(data.fill_(0))

# from safetensors.torch import load_file
# model_path = r'D:\seekoo\SD\sd-scripts\result\renwu_full.safetensors'
# lora_sd = load_file(model_path)
# print()