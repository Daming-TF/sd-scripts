import os.path
import subprocess
import time
import numpy as np
import argparse
import random


"""
    Layer-wise：
    可以根据locked layer固定某部分的权重，对不锁的的部分根据class_blocks_ratios加权融合
"""
# TE_LOCKEDKEYS = []
# TE_LOCKEDKEYS = ['encoder_layers']
# UNE_LOCKEDKEYS = ['down_blocks', 'up_blocks']
# UNE_LOCKEDKEYS = ['down_blocks_0', 'down_blocks_1', 'down_blocks_2', 'down_blocks_3', 'mid_block', 'up_blocks_0', '']
TE_LOCKEDKEYS = ['encoder_layers_0', 'encoder_layers_1', 'encoder_layers_2', 'encoder_layers_3', 'encoder_layers_4',
                 'encoder_layers_5', 'encoder_layers_6', 'encoder_layers_7', 'encoder_layers_8', 'encoder_layers_9',
                 'encoder_layers_10']
UNE_LOCKEDKEYS = ['down_blocks_0', 'down_blocks_1', 'down_blocks_2', 'down_blocks_3', 'mid_block',
                  'up_blocks_0', 'up_blocks_1', 'up_blocks_2', 'up_blocks_3']


def process_run(args):
    # ratios_list = np.arange(0, 1.5, 0.5).tolist() if args.ratios is None \
    #     else np.arange(float(args.ratios[0]), float(args.ratios[1]), float(args.ratios[2])).tolist()  # [0, 0.5, 1]
    # ratios_groups = list(itertools.product(ratios_list, repeat=len(args.block_names)))[::-1]

    # get locked key
    locked_keys = ' '.join(map(str, TE_LOCKEDKEYS + UNE_LOCKEDKEYS))

    # exp2
    group = [random.random() for _ in range(len(TE_LOCKEDKEYS + UNE_LOCKEDKEYS))]
    # total = sum(random_list)
    # group = [num / total for num in random_list]

    # exp1
    # group = []
    # for i in range(len(TE_LOCKEDKEYS + UNE_LOCKEDKEYS)):
    #     if i % 2 == 0:  # 偶数
    #         group.append(args.weights)
    #     else:
    #         group.append(1-args.weights)

    assert len(group) == 20

    for seed in range(args.seed_freq):
        # get ratios
        # group = [args.weights]
        ratios_group = ' '.join(map(str, group))

        # get save path
        # weight_filename = f"{'{:.3f}'.format(args.weights)}"
        weight_filename = '-'.join(map(str, ['{:.3f}'.format(weight) for weight in group]))
        save_path = os.path.join(args.save_dir, weight_filename)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, str(seed))
        if os.path.exists(save_path) and len(os.listdir(save_path)) == 4:
            continue
        else:
            os.makedirs(save_path, exist_ok=True)

        cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models ../result/best_with_te/renwu_all.safetensors ../result/best_with_te/colorful_world_all.safetensors --save_dir={save_path} --remark_info {seed} --locked_keys {locked_keys} --class_blocks_ratios {ratios_group} --prompt_txt ../config/prompt_webui_script_align.txt --seed {seed}"

        for index in range(3):
            if subprocess.call(cmd) == 0:
                break

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=float,
    )
    parser.add_argument(
        "--seed_freq", type=int,
    )
    parser.add_argument(
        "--save_dir", type=str, default=r'E:\Data\test\ScriptMerge-2'
    )
    args = parser.parse_args()
    process_run(args)

