"""
图像数据增强(augmentation)操作
"""

import random
import numpy as np
import cv2

def random_rotate_img(img, min_angle, max_angle):
    """
    图像旋转
    random rotation an image

    :param img:         image to be rotated
    :param min_angle:   min angle to rotate
    :param max_angle:   max angle to rotate
    :return:            image after random rotated

    """

    if not isinstance(img, list):
        img = [img]

    angle = random.randint(min_angle, max_angle)
    center = (img[0].shape[0] / 2, img[0].shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, rot_matrix,dsize=img_inst.shape[:2],
                                  borderMode=cv2.BORDER_CONSTANT)
        res.append(img_inst)
    if len(res) == 0:
        res = res[0]
    return res


def random_flip_img(img):
    '''
	图像平移
    random flip image,both on horizontal and vertical
    :param img:                 image to be flipped
    :return:                    image after flipped
    '''
    flip_val = 0
    if not isinstance(img, list):
        res = cv2.flip(img, flip_val)  # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res

if __name__ == '__main__':
    imagePath = './data/chaper3_img_01.png'
    # 读取dicom文件的元数据(dicom tags)
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    res_rotate = random_rotate_img(img, 30, 90)
    cv2.imwrite('./temp_dir/chapter3_rotate_img.png', res_rotate[0])
    res_flip = random_flip_img(img)
    cv2.imwrite('./temp_dir/chapter3_flip_img.png', res_flip)
