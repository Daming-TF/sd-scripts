import os
import random
import subprocess
import time
import numpy as np
import argparse
import itertools
import os


"""
    尝试融合9个LoRA，并把colorful world放到首位，初始每个LoRA基础得分为随机得分1~5，colorful world初始得分为1慢慢增加到30，每次增加5分
"""
RetryMAX = 5
ModelName = 'colorful_world_all'
CoreModelScore = 1


def recored_info(txt_path, model_scores, models, ratios):
    model_scores_info = '\t'.join(map(str,model_scores))
    model_name_list = [os.path.basename(model).split('.')[0] for model in models]
    with open(txt_path, 'a', encoding='utf-8') as file:
        file.write(f'ModelScore:\t'+model_scores_info+'\n')
        for ratio, model in zip(model_name_list, ratios):
            file.write(f'{model}:\t{str(ratio)}\n')


def move_element_to_position(lst, element, position):
    if element in lst:
        lst.remove(element)
        lst.insert(position, element)
    else:
        print(f"element >> {element} << is not in the list")


def process_run(args):
    # get model path list and let core model to the first position
    models = [os.path.join(args.ckpt_path, model_name) for model_name in os.listdir(args.ckpt_path)
              if model_name.endswith('.safetensors') and 'colorful_rhythm' not in model_name ]
    targe_model = [model_path for model_path in models if ModelName in model_path][0]
    move_element_to_position(models, targe_model, 0)
    models_txt = ' '.join(models)

    # init all models score and get ratios
    model_scores = [random.randint(1, 5) for _ in range(len(models)-1)]
    core_model_score = args.core_model_score if args.core_model_score is None else CoreModelScore
    model_scores.insert(0, core_model_score)


    # iter accored seed
    for i in range(int(40 / 5)):
        core_model_score = 10 + core_model_score if i != 0 else core_model_score
        model_scores[0] = core_model_score
        ratios = [score / sum(model_scores) for score in model_scores]
        ratios_txt = ' '.join(map(str, ratios))

        # get save dir
        save_dir = os.path.join(args.save_dir, f'CoreModelScore-{str(core_model_score)}')
        os.makedirs(save_dir, exist_ok=True)

        # check
        assert len(models) == len(ratios)

        # record import info
        txt_path = os.path.join(save_dir, 'RecordInfo.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as file:
                info = file.readline()
                model_scores = [int(score) for score in info.split('\t')[1:]]

        recored_info(txt_path, model_scores, models, ratios)

        for seed in range(args.seed_freq):
            # get save path
            save_path = os.path.join(save_dir, str(seed))
            if os.path.exists(save_path) and len(os.listdir(save_path)) == 4:
                continue
            else:
                os.makedirs(save_path, exist_ok=True)

            #
            cmd = rf"D:\tool\Anaconda3\envs\LoRA\python.exe  D:\seekoo\SD\sd-scripts\my_tool\merge_param.py --models {models_txt}  --ratios {ratios_txt} --prompt_txt ../config/prompt_webui_script_align.txt --seed {seed} --remark_info {seed} --save_dir {save_path}"

            # avoid https error
            for index in range(RetryMAX):
                if subprocess.call(cmd) == 0:
                    break

            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str,
    )
    parser.add_argument(
        "--seed_freq", type=int, default=3
    )
    parser.add_argument(
        "--save_dir", type=str, default=r'E:\Data\test\ScriptMerge-2'
    )
    parser.add_argument(
        "--core_model_score", type=int, default=None
    )
    args = parser.parse_args()
    process_run(args)

