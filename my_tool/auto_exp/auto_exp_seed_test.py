import os
import random
import subprocess
import time
import numpy as np
import argparse
import itertools
import os


"""
    Layer-wise复杂版：
    对locked block name所有层随机采样归一化融合
"""
RETRYMAX = 3


def generate_lists(sample_freq, blocks_num, models_num):
    result = []
    for _ in range(sample_freq):
        mid = []
        for _ in range(blocks_num):
            random_list = [random.random() for _ in range(models_num)]
            total = sum(random_list)
            normalized_list = [num / total for num in random_list]  # 归一化处理，使列表元素相加为1
            mid.append(normalized_list)
        result.append(mid)
    return result


def process_run(args):
    model_names = [dir.split('.')[0]if 'color' in dir.split('.')[0] else dir.split('.')[0][:5]
                   for dir in os.listdir(args.ckpt_path)]
    model_paths = [os.path.join(args.ckpt_path, dir) for dir in os.listdir(args.ckpt_path)]

    record_txt_path = fr'E:\Data\test\Multi-LoRA-{len(model_names)}\record.txt'
    if not os.path.exists(os.path.dirname(record_txt_path)):
        os.mkdir(os.path.dirname(record_txt_path))
    # make ratios_groups
    # dim0:sample freq;  dim1:locked keys数量;  dim2:对应models数量

    for i in range(10):
        seed = random.randint(5, 100)
        ratios_groups = generate_lists(args.sample_freq, len(args.locked_keys), len(model_names))

        for index, ratios_group in enumerate(ratios_groups):      # ratios_group (shape): {blocks_num, lora num}
            locked_keys = ' '.join(map(str, args.locked_keys))
            ratios = ' '.join(map(str, np.array(ratios_group).reshape(-1)))
            models = ' '.join(map(str, model_paths))

            sample_index = index+args.start_index

            remark_info = f'Index@{index+args.start_index}@@'
            remark_info += 'Models@' + '@'.join(map(str, model_names)) + '@@'
            for _, (name, ratio) in enumerate(zip(args.locked_keys, ratios_group)):
                if _ == 0:
                    remark_info += '@'
                remark_info += f'{name}@'
                for i, r in enumerate(ratio):
                    remark_info += f'{model_names[i]}@{r}' + '@'
                remark_info += '@@'

            with open(record_txt_path, 'a', encoding='utf-8') as file:
                for info in remark_info.split('@@'):
                    file.write(info.replace('@', '\t') + '\n')

            cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models {models} --save_dir=E:\Data\test\Multi-LoRA-{len(model_names)}\Seed_{seed}_Index_{sample_index} --remark_info seed{seed}_{sample_index} --locked_keys {locked_keys} --multi_lora_ratios {ratios} --seed {seed}"

            for index in range(RETRYMAX):
                if subprocess.call(cmd) == 0:
                    break

            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locked_keys", nargs="*",
        default=None,)
    parser.add_argument(
        "--sample_freq", type=int,
    )
    parser.add_argument(
        "--ckpt_path", type=str,
    )
    parser.add_argument(
        "--start_index", type=int,
        default=0
    )
    parser.add_argument(
        "--seed", type=int, default=int("0607105102")
    )

    args = parser.parse_args()
    process_run(args)

