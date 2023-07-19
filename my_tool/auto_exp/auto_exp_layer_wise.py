import os.path
import subprocess
import time
import numpy as np
import argparse
import itertools


"""
    Layer-wise：
    可以根据locked layer固定某部分的权重，对不锁的的部分根据class_blocks_ratios加权融合
"""


def process_run(args):
    block_names = ' '.join(map(str, args.block_names))

    # ratios_groups = [[0.5, 0.5, 0.5]]
    ratios_list = np.arange(0, 1.5, 0.5).tolist() if args.ratios is None \
        else np.arange(float(args.ratios[0]), float(args.ratios[1]), float(args.ratios[2])).tolist()      # [0, 0.5, 1]

    ratios_groups = list(itertools.product(ratios_list, repeat=len(args.block_names)))[::-1]

    assert len(args.block_names) == len(ratios_groups[0])

    for group in ratios_groups:
        ratios_group = ' '.join(map(str, group))
        # group_name = '_'.join(map(str, group))

        if filename is None:
            filename = ''
            key=''
            for name, ratio in zip(args.block_names, group):
                if 'down' in name:
                    key = 'D'+name.split('_')[-1]
                elif 'mid' in name:
                    key = 'M'+name.split('_')[-1]
                elif 'up' in name:
                    key = 'U'+name.split('_')[-1]
                filename += f'{key}_{ratio}_'
        else:
            filename = args.save_name

        # if os.path.exists(fr"E:\Data\test\LayerWise_{block_name}_{group_name}"):
        #     continue
        cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models ../result/renwu_wo_te.safetensors ../result/healing_wo_te.safetensors --save_dir=E:\Data\test\LayerWise_{filename} --remark_info {filename} --locked_keys {block_names} --class_blocks_ratios {ratios_group}"

        for index in range(3):
            if subprocess.call(cmd) == 0:
                break

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block_names", nargs="*",
        default=None,)
    parser.add_argument(
        "--ratios", nargs="*",
        default=None, )
    parser.add_argument(
        "--save_name", default=None,
    )
    parser.add_argument(
        "--seed", type=int,
    )
    args = parser.parse_args()
    process_run(args)

