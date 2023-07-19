import subprocess
import time
import numpy as np
import argparse


def process_run(args):
    block_names = args.block_names
    ratios_list = np.arange(0, 1, 0.2).tolist() if args.ratios is None \
        else np.arange(float(args.ratios[0]), float(args.ratios[1]), 0.2).tolist()

    commands = [
        r"d:"
        r"cd D:\seekoo\SD\sd-scripts\my_tool",
    ]
    for block_name in block_names:
        for ratios in ratios_list:
            ratios = round(round((ratios), 1)*10/10, 1)
            a = round(round((1-ratios), 1)*10/10, 1)
            i = rf"D:\tool\Anaconda3\envs\LoRA\python.exe ./merge_param.py --models ../result/renwu_wo_te.safetensors ../result/healing_wo_te.safetensors --ratios {ratios} {a} --save_dir=E:\Data\test\{block_name}_{ratios}renwu_{a}healing --remark_info {block_name}_{ratios}renwu_{a}healing --locked_keys {block_name}"
            commands.append(i)

            for cmd in commands:
                if subprocess.call(cmd) == 0:
                    break
                else:
                   if subprocess.call(cmd) == 0:
                       break
                   else:
                       exit(1)
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block_names", nargs="*",
        default=None,)
    parser.add_argument(
        "--ratios", nargs="*",
        default=None, )
    args = parser.parse_args()
    process_run(args)

