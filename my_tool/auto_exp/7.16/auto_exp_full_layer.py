import os
import random
import subprocess
import time
import numpy as np
import argparse
import itertools
import os


"""
    主要用于研究两个LoRA数据集融合训练效果如何，分别从epoch，promot，seed三个维度观察
"""
RETRYMAX = 5


def process_run(args):
    for seed in range(args.seed_freq):
        ratios_group = [1]
        ratios = ' '.join(map(str, np.array(ratios_group[0]).reshape(-1)))

        # file_name = 'Vector-BeautifulAsCrystal'
        file_name = os.path.basename(args.models).split('.')[0]
        save_dir = os.path.join(args.save_dir, file_name)
        os.makedirs(save_dir, exist_ok=True)
        # save_dir = os.path.join(save_dir, args.epoch)
        # os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, str(seed))
        if os.path.exists(save_path) and len(os.listdir(save_path)) == 4:
            continue
        else:
            os.makedirs(save_path, exist_ok=True)

        cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models {args.models}  --ratios 1 --prompt_txt ../config/prompt_webui_script_align.txt --seed {seed} --remark_info {seed} --save_dir {save_path}"

        for index in range(RETRYMAX):
            if subprocess.call(cmd) == 0:
                break

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_freq", type=int, default=3
    )
    parser.add_argument(
        "--save_dir", type=str, default=r'E:\Data\test\ScriptMerge-2'
    )
    parser.add_argument(
        "--epoch", type=None
    )
    parser.add_argument(
        "--models", type=str
    )
    args = parser.parse_args()
    process_run(args)

