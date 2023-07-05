import os
from pic_concat import concatenate_images
import math


def main():
    data_dir = r'E:\Data\test'
    pic_dirs = [os.path.join(data_dir, dir) for dir in os.listdir(data_dir)]
    pic_names = []
    for name in os.listdir(os.path.join(data_dir, pic_dirs[0])):
        if os.path.splitext(name)[1] == '.png':
            pic_names.append(name)

    for name in pic_names:
        img_paths = [os.path.join(dir, name) for dir in pic_dirs]
        name.replace(',', '_')
        save_path = os.path.join(r'E:\Data\test', name)

        concatenate_images(img_paths, save_path)


if __name__ == '__main__':
    main()
