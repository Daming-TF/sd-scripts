import os
from pic_concat import concatenate_images
import math
from tqdm import tqdm

# KEY1 = 'U0_1.0'    # 'mid_block'
# KEY2 = 'U1_1.0'    # 'mid_block'
# KEY3 = 'U3_1.0'
KEY = ''
dir_file = 'pic'


def main():
    data_dir = r'E:\Data\test\ScriptMerge-2\renwu-colorfulworld\no_people_prompt\colorfulworld-renwu\colorfulworld-0.75'
    # pic_dirs = [os.path.join(data_dir, dir) for dir in os.listdir(data_dir)
    #             if KEY1 in dir and KEY2 in dir and KEY3 in dir]
    pic_dirs = [os.path.join(data_dir, dir) for dir in os.listdir(data_dir)
                if KEY in dir and dir != dir_file and '.' not in dir]

    pic_names = []
    for name in os.listdir(os.path.join(data_dir, pic_dirs[0])):
        if os.path.splitext(name)[1] == '.png':
            pic_names.append(name)

    save_dir = os.path.join(data_dir, dir_file)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in tqdm(range(len(pic_names))):
        name = pic_names[i]
        img_paths = [os.path.join(dir, name) for dir in pic_dirs]
        # name.replace(',', '_')
        save_path = os.path.join(save_dir, KEY+name)

        concatenate_images(img_paths, save_path, row=1, col=6)


if __name__ == '__main__':
    main()
