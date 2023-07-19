import os
from tqdm import tqdm


def main():
    txt_dir = r'E:\Data\TrainData\xinhaicheng'
    txt_paths = [os.path.join(txt_dir, filename) for filename in os.listdir(txt_dir) if filename.endswith('.txt')]
    for txt_path in tqdm(txt_paths):
        with open(txt_path, 'r')as file:
            lines = file.readline()
            if 'xinhaicheng' != lines[:11]:
                exit(0)
            new_content = lines[13:]

        with open(txt_path, 'w')as file:
            file.write(new_content)
        # print("文档内容已成功覆盖！")


if __name__ == '__main__':
    main()
