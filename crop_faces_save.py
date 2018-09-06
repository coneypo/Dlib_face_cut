# created at 2018-01-22
# updated at 2018-09-06

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_cut

import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv

# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 读取图像的路径
path_read = "images/"
img = cv2.imread(path_read+"test_faces_1.jpg")

# 用来存储生成的单张人脸的路径
path_save = "crop_faces_save_folder/"

# Dlib 检测
faces = detector(img, 1)

print("人脸数：", len(faces))

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
