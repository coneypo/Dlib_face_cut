# created at 2018-01-22
# updated at 2018-09-29

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_cut

import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv
import os

# 读取图像的路径
path_read = "data/images/faces_for_test/"
img = cv2.imread(path_read+"test_faces_3.jpg")

# 用来存储生成的单张人脸的路径
path_save = "data/images/faces_separated/"


# Delete old images
def clear_images():
    imgs = os.listdir(path_save)

    for img in imgs:
        os.remove(path_save + img)

    print("clean finish", '\n')


clear_images()


# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')


# Dlib 检测
faces = detector(img, 1)

print("人脸数：", len(faces), '\n')

for k, d in enumerate(faces):

    # 计算矩形大小
    # (x,y), (宽度width, 高度height)
    pos_start = tuple([d.left(), d.top()])
    pos_end = tuple([d.right(), d.bottom()])

    # 计算矩形框大小
    height = d.bottom()-d.top()
    width = d.right()-d.left()

    # 根据人脸大小生成空的图像
    img_blank = np.zeros((height, width, 3), np.uint8)

    for i in range(height):
        for j in range(width):
                img_blank[i][j] = img[d.top()+i][d.left()+j]

    # cv2.imshow("face_"+str(k+1), img_blank)

    # 存在本地
    print("Save to:", path_save+"img_face_"+str(k+1)+".jpg")
    cv2.imwrite(path_save+"img_face_"+str(k+1)+".jpg", img_blank)

