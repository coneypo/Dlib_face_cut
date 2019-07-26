# created at 2018-01-22
# updated at 2019-07-26

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_cut

import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv

# Dlib 检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

# 读取图像
path = "data/images/faces_for_test/"
img = cv2.imread(path+"test_faces_1.jpg")

# Dlib 检测
faces = detector(img, 1)

print("人脸数 / faces in all:", len(faces), "\n")

# 记录人脸矩阵大小
height_max = 0
width_sum = 0

# 计算要生成的图像 img_blank 大小
for face in faces:

    # 计算矩形大小
    # (x,y), (宽度width, 高度height)
    pos_start = tuple([face.left(), face.top()])
    pos_end = tuple([face.right(), face.bottom()])

    # 计算矩形框大小
    height = face.bottom()-face.top()
    width = face.right()-face.left()

    # 处理宽度
    width_sum += width

    # 处理高度
    if height > height_max:
        height_max = height
    else:
        height_max = height_max

# 绘制用来显示人脸的图像的大小
print("窗口大小 / The size of window:"
      , '\n', "高度 / height:", height_max
      , '\n', "宽度 / width: ", width_sum)

# 生成用来显示的图像
img_blank = np.zeros((height_max, width_sum, 3), np.uint8)

# 记录每次开始写入人脸像素的宽度位置
blank_start = 0

# 将人脸填充到 img_blank
for face in faces:

    height = face.bottom()-face.top()
    width = face.right()-face.left()

    # 填充
    for i in range(height):
        for j in range(width):
                img_blank[i][blank_start+j] = img[face.top()+i][face.left()+j]
    # 调整图像
    blank_start += width

cv2.namedWindow("img_faces")#, 2)
cv2.imshow("img_faces", img_blank)
cv2.waitKey(0)