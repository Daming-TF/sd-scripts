import subprocess
import time
import numpy as np


if __name__ == "__main__":
    ratios_list = np.arange(1, 1.1, 0.1).tolist()

    commands = [
        r"d:"
        r"cd D:\seekoo\SD\sd-scripts\my_tool",
    ]
    for ratios in ratios_list:
        ratios = round(round((ratios), 1)*10/10, 1)
        a = round(round((1-ratios), 1)*10/10, 1)
        i = rf"D:\tool\Anaconda3\envs\LoRA\python.exe \
        merge_param.py \
        --models ../result/renwu_wo_te.safetensors ../result/healing_wo_te.safetensors \
        --ratios {ratios} {a} \
        --save_dir=E:\Data\test\merge_param_{ratios}renwu_{a}healing \
        --remark_info MergeParam_renwu_{ratios}-healing_{a}"

        commands.append(i)

        for cmd in commands:
            if subprocess.call(i) == 0:
                break
            else:
               if subprocess.call(i) == 0:
                   break
               else:
                   exit(1)
        time.sleep(1)
