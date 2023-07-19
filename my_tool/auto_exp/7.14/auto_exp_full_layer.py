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
    两个loRA，全网络结果融合，Unet&Te：
"""
RETRYMAX = 5


def process_run(args):
    for seed in range(args.seed_freq):
        ratios_group = args.ratios
        ratios = ' '.join(map(str, np.array(ratios_group).reshape(-1)))

        models = ' '.join(args.models)

        # weight_filename = f"{'{:.3f}'.format(ratios_group[0])}-{'{:.3f}'.format(ratios_group[1])}"
        # weight_filename = 'colourworld-renwu'
        # save_path = os.path.join(args.save_dir, weight_filename)
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, str(seed))
        if os.path.exists(save_path) and len(os.listdir(save_path)) == 4:
            continue
        else:
            os.makedirs(save_path, exist_ok=True)

        cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models {models} --ratios {ratios} --prompt_txt ../config/prompt_nopeople.txt --seed {seed} --remark_info {seed} --save_dir {save_path}"

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
        "--models", type=str, nargs='*',
    )
    parser.add_argument(
        "--ratios", type=float, nargs='*',
    )
    args = parser.parse_args()
    process_run(args)

