import subprocess
import time
import numpy as np


# ['colorful_world', 'fight', 'flat_girl', 'vector', 'beautiful_as_crystal', 'colorful_rhythm', 'oldschool', 'parallel']
# 'renwu', 'healing', 'xinhaicheng', 'colorful_world', 'fight', 'flat_girl', 'vector', 'beautiful_as_crystal', 'colorful_rhythm', 'oldschool'
if __name__ == "__main__":
    data_names =['exp_data_mix-Colorfulrhythm-Xinhaicheng', 'exp_data_mix-Oldschool-Healing',
                 'exp_data_mix-Vector-BeautifulAsCrystal']

    for name in data_names:
        i = rf"D:\tool\Anaconda3\envs\LoRA\python.exe train_network.py --learning_rate=1e-4 --max_train_epochs=100 --save_every_n_epochs=10 --sample_every_n_epochs=10 --network_module=networks.lora --text_encoder_lr=5e-5 --noise_offset=0.1 --lr_scheduler_num_cycles=1 --network_dim=128 --network_alpha=128 --max_data_loader_n_workers=0 --sample_prompts=./config/prompt.txt --pretrained_model_name_or_path=./checkpoint/v1-5-pruned.ckpt --output_dir=./result --dataset_config=./config/{name}.toml --output_name={name} --network_args 'conv_dim=4' 'conv_alpha=4'"

        if subprocess.call(i) == 0:
            continue
        else:
            exit(1)

        time.sleep(5)

