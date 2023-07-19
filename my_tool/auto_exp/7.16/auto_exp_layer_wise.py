import os.path
import subprocess
import time
import numpy as np
import argparse
import random


"""
    探究encoder是否决定生成图片中人物的姿态
"""
# TE_LOCKEDKEYS = []
TE_LOCKEDKEYS = ['encoder_layers']
UNE_LOCKEDKEYS = ['down_blocks', 'mid_block']
# UNE_LOCKEDKEYS = ['down_blocks_0', 'down_blocks_1', 'down_blocks_2', 'down_blocks_3', 'mid_block', 'up_blocks_0', '']
# TE_LOCKEDKEYS = ['encoder_layers_0', 'encoder_layers_1', 'encoder_layers_2', 'encoder_layers_3', 'encoder_layers_4',
#                  'encoder_layers_5', 'encoder_layers_6', 'encoder_layers_7', 'encoder_layers_8', 'encoder_layers_9',
#                  'encoder_layers_10']
# UNE_LOCKEDKEYS = ['down_blocks_0', 'down_blocks_1', 'down_blocks_2', 'down_blocks_3', 'mid_block',
#                   'up_blocks_0', 'up_blocks_1', 'up_blocks_2', 'up_blocks_3']


def process_run(args):
    # ratios_list = np.arange(0, 1.5, 0.5).tolist() if args.ratios is None \
    #     else np.arange(float(args.ratios[0]), float(args.ratios[1]), float(args.ratios[2])).tolist()  # [0, 0.5, 1]
    # ratios_groups = list(itertools.product(ratios_list, repeat=len(args.block_names)))[::-1]

    # get locked key
    locked_keys = ' '.join(map(str, TE_LOCKEDKEYS + UNE_LOCKEDKEYS))

    for seed in range(args.seed_freq):
        ratios = [args.class_blocks_ratios]*len(TE_LOCKEDKEYS + UNE_LOCKEDKEYS)
        class_blocks_ratios = ' '.join(map(str, ratios))

        # get save path
        save_dir = os.path.join(args.save_dir,
                                 f'colorfulrhythm-{str(args.class_blocks_ratios)}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, str(seed))
        if os.path.exists(save_path) and len(os.listdir(save_path)) == 4:
            continue
        else:
            os.makedirs(save_path, exist_ok=True)

        cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models ../result/best_with_te/colorful_rhythm_all.safetensors {args.models} --save_dir={save_path} --remark_info {seed} --locked_keys {locked_keys} --class_blocks_ratios {class_blocks_ratios} --prompt_txt ../config/prompt_webui_script_align.txt --seed {seed}"

        for index in range(3):
            if subprocess.call(cmd) == 0:
                break

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_freq", type=int,
    )
    parser.add_argument(
        "--class_blocks_ratios", type=float,
    )
    parser.add_argument(
        "--save_dir", type=str, default=r'E:\Data\test\ScriptMerge-2'
    )
    parser.add_argument(
        '--models', type=str,
    )
    args = parser.parse_args()
    process_run(args)

