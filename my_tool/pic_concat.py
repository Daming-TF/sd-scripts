import os
import cv2
import numpy as np
import math


def concatenate_images(img_paths, save_path, row=4, col=4):
    # row = int(math.sqrt(len(img_paths)))
    # col = row+1

    h, w, _ = cv2.imread(img_paths[0]).shape
    result_width = w * col
    result_height = h * row

    result_image = np.zeros((result_height, result_width, 3), dtype=np.uint8)
    cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    for i, img_path in enumerate(img_paths):
        image = cv2.imread(img_path)

        if image is None:
            continue

        row_index = i // col
        column_index = i % col
        left = column_index * w
        top = row_index * h

        result_image[top:top+h, left:left+w, :] = cv2.resize(image, (w, h))

    cv2.imwrite(save_path, result_image)


if __name__ in '__main__':
    image_dir = r'D:\seekoo\inpainting\Adobe'
    output_path = r'D:\seekoo\inpainting\Adobe\14.png'

    img_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)
                 if os.path.splitext(name)[1] == '.png' and '14-' in name]
    concatenate_images(img_paths, output_path, row=2, col=2)

