import os
import random
import subprocess
import time
import numpy as np
import argparse
import itertools
import os


"""
    Layer-wise复杂版
    两个loRA，全网络结果融合，Unet$Te：
"""
RETRYMAX = 5


def generate_lists(sample_freq, models_num):
    result = []
    for _ in range(sample_freq):
        random_list = [random.random() for _ in range(models_num)]
        total = sum(random_list)
        normalized_list = [num / total for num in random_list]  # 归一化处理，使列表元素相加为1
        result.append(normalized_list)
    return result


def process_run(args):
    # make ratios_groups
    # dim0:sample freq;  dim1:models数量
    ratios_groups = generate_lists(args.sample_freq, 2)

    for index, ratios_group in enumerate(ratios_groups):      # ratios_group (shape): {blocks_num, lora num}
        for seed in range(args.seed_freq):
            ratios_group = [1, 0]
            ratios = ' '.join(map(str, np.array(ratios_group).reshape(-1)))

            weight_filename = f"{'{:.3f}'.format(ratios_group[0])}-{'{:.3f}'.format(ratios_group[1])}"
            save_path = os.path.join(args.save_dir, weight_filename)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, str(seed))
            if os.path.exists(save_path) and len(os.listdir(save_path)) == 4:
                continue
            else:
                os.makedirs(save_path, exist_ok=True)

            cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models ../result/best_with_te/oldschool_all.safetensors ../result/best_with_te/healing_all.safetensors  --ratios {ratios} --prompt_txt ../config/prompt_webui_script_align.txt --seed {seed} --remark_info {seed} --save_dir {save_path}"

            for index in range(RETRYMAX):
                if subprocess.call(cmd) == 0:
                    break

            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_freq", type=int, default=5
    )
    parser.add_argument(
        "--seed_freq", type=int, default=3
    )
    parser.add_argument(
        "--save_dir", type=str, default=r'E:\Data\test\ScriptMerge-2'
    )
    args = parser.parse_args()
    process_run(args)

