import os.path
import subprocess
import time
# import numpy as np
import argparse
from tqdm import tqdm


MAX = 5
def process_run(args):
    # sample
    lora_paths = [os.path.join(args.ckpt_path, ckpt_name)for ckpt_name in os.listdir(args.ckpt_path)
                       if ckpt_name.endswith('.safetensors')]
    # print(lora_paths)
    # assert len(args.block_names) == len(ratios_groups[0])

    for lora_path in tqdm(lora_paths):
        lora_name = os.path.basename(lora_path).split('.')[0]
        cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models {lora_path} --ratios 1 --save_dir=E:\Data\test\OverFitting\{lora_name} --remark_info {lora_name} --prompt_txt D:\seekoo\SD\sd-scripts\config\prompt_debug.txt"

        run_state = False
        for index in range(MAX):
            if subprocess.call(cmd) == 0:
                run_state = True
                break

        if run_state is False:
            ValueError("HTTPs Conection may occur Error!!")

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_path', type=str
    )

    args = parser.parse_args()
    process_run(args)
