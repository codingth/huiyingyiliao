#模型预测的相关功能

from NN_train import get_unet
from OptimizeModel import get_3dnnnet, stack_2dcube_to_3darray, prepare_image_for_net3D, MEAN_PIXEL_VALUE
import glob
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from skimage import morphology
import os

CHANNEL_COUNT = 1
_3DCNN_WEIGHTS = './model/3dcnn.hd5'
UNET_WEIGHTS = './model/unet.hd5'
THRESHOLD = 2
BATCH_SIZE = 1


# 获取unet预测结果的中心点坐标(x,y)
def unet_candidate_dicom(unet_result_path):
    centers = []
    image_t = cv2.imread(unet_result_path, cv2.IMREAD_GRAYSCALE)
    # Thresholding(阈值化)
    image_t[image_t < THRESHOLD] = 0
    image_t[image_t > 0] = 1
    # dilation（扩张）
    selem = morphology.disk(1)
    image_eroded = morphology.binary_dilation(image_t, selem=selem)
    label_im, nb_labels = ndimage.label(image_eroded)

    for i in range(1, nb_labels + 1):
        blob_i = np.where(label_im == i, 1, 0)
        print(blob_i)
        mass = center_of_mass(blob_i)
        y_px = int(round(mass[0]))
        x_px = int(round(mass[1]))
        centers.append([y_px, x_px])
    return centers


# 数据输入网络之前先进行预处理
def prepare_image_for_net(img):
    img = img.astype(np.float)
    img /= 255.
    if len(img.shape) == 3:
        img = img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])
    else:
        img = img.reshape(1, img.shape[-2], img.shape[-1], 1)
    return img


# unet模型的预测代码
def unet_predict(imagepath,Newdir1):
    model = get_unet()
    model.load_weights(UNET_WEIGHTS)
    # read png and ready for predict
    images = []
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    images.append(img)
    for index, img in enumerate(images):
        img = prepare_image_for_net(img)
        images[index] = img
    images3d = np.vstack(images)
    y_pred = model.predict(images3d, batch_size=BATCH_SIZE)
    print(len(y_pred))
    count = 0
    for y in y_pred:
        y *= 255.
        y = y.reshape((y.shape[0], y.shape[1])).astype(np.uint8)
        cv2.imwrite(Newdir1, y)
        count += 1


# 3dcnn模型的预测代码
def _3dcnn_predict(imagepath):
    cube_image = stack_2dcube_to_3darray(imagepath, 4, 8, 32)
    img3d = prepare_image_for_net3D(cube_image, MEAN_PIXEL_VALUE)
    model = get_3dnnnet(load_weight_path='./model/3dcnn.hd5')
    result = model.predict(img3d, batch_size=BATCH_SIZE, verbose=1)
    print('3dcnn result: ', result)

def plot_one_box(img, coord, label=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.矩形线条粗细
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    # color = [random.randint(0, 255) for _ in range(3)]
    color = [0,0,255]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))# 中心点，宽高
    # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩形
    # img是原图（x，y）是矩阵的左上点坐标（x+w，y+h）是矩阵的右下点坐标
    # （0,255,0）是画线对应的rgb颜色2是所画的线的宽度
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    # 在矩形框上显示出类别
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1)  # filled
    #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    savepath = r'E:\Pycharm-community-2019.1\WorkSpace\Medical_CT\nodule_detection-master\data\11'
    for files in os.listdir(savepath):
        Olddir = os.path.join(savepath, files)
        # Newdir = os.path.join('D:/TH',files)
        Newdir1 = os.path.join('D:/123',files)
        img_ori = cv2.imread(Olddir)
        unet_predict(Olddir,Newdir1)
        centers = unet_candidate_dicom(Newdir1)
        print('y, x', centers)
        # for i in range(len(centers)):
        #     box = [centers[i][1]-5.5,centers[i][0]-5.5,centers[i][1]+5.5,centers[i][0]+5.5]
        #     plot_one_box(img_ori, box)
        # cv2.imwrite(Newdir,img_ori)
        # cv2.imshow('Detection result', img_ori)

        # _3dcnn_predict('./data/chapter6/true_positive_nodules.png')
        # _3dcnn_predict('./data/chapter6/false_positive_nodules.png')
        # cv2.waitKey(0)