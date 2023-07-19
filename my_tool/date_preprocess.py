import os
import random
import shutil
from tqdm import tqdm


def get_random_files(path, num_files):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_list.append(os.path.join(root, file))

    random_files = random.sample(file_list, num_files)
    return random_files


if __name__ == '__main__':
    source_dir = r"E:\Data\TrainData\xinhaicheng"
    target_dir = r"E:\Data\TrainData\exp_data_mix-Colorfulrhythm-Xinhaicheng"

    # 获取随机文件路径列表
    random_file_paths = get_random_files(source_dir, 100)

    # 打印随机文件路径列表
    for file_path in tqdm(random_file_paths):
        source_dir, file_suffix = os.path.splitext(file_path)
        source_dir = os.path.dirname(source_dir)
        file_name = os.path.basename(file_path).split(file_suffix)[0]
        for suffix in ['.txt', '.png']:
            source_path = os.path.join(source_dir, file_name+suffix)
            target_path = os.path.join(target_dir, file_name+suffix)
            shutil.copy(source_path, target_path)

