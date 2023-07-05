import subprocess
import time
import numpy as np


if __name__ == "__main__":
    data_names =['xinhaicheng', 'renwu', 'healing']

    commands = [
        r"d:"
        r"cd D:\seekoo\SD\sd-scripts",
    ]
    for name in data_names:
        i = rf"D:\tool\Anaconda3\envs\LoRA\python.exe train_network.py --learning_rate=1e-4 --max_train_epochs=100 --save_every_n_epochs=10 --sample_every_n_epochs=10 --network_module=networks.lora --text_encoder_lr=5e-5 --noise_offset=0.1 --lr_scheduler_num_cycles=1 --network_dim=128 --network_alpha=128 --max_data_loader_n_workers=0 --sample_prompts=./config/prompt.txt --network_train_unet_only --pretrained_model_name_or_path=./checkpoint/dreamshaper_7.safetensors --output_dir=./result --dataset_config=./config/{name}.toml --output_name=dreamshaper_{name}_wo_te --network_args 'conv_dim=4' 'conv_alpha=4'"
        commands.append(i)

        for cmd in commands:
            if subprocess.call(i) == 0:
                break

        time.sleep(5)
