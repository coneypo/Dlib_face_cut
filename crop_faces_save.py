# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_cut

import dlib
import numpy as np
import cv2
import os

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

# 读取图像的路径
path_read = "data/images/faces_for_test/"
img = cv2.imread(path_read + "test_faces_10.jpg")

# 用来存储生成的单张人脸的路径
path_save = "data/images/faces_separated/"


def mkdir_for_save_images():
    if not os.path.isdir(path_save):
        os.mkdir(path_save)


def clear_images():
    img_list = os.listdir(path_save)
    for img in img_list:
        os.remove(path_save + img)


def main():
    # 新建文件夹, 清理存下来的图像文件
    mkdir_for_save_images()
    clear_images()

    faces = detector(img, 1)

    print("人脸数 / faces in all:", len(faces), '\n')

    for num, face in enumerate(faces):

        # 计算矩形框大小
        height = face.bottom() - face.top()
        width = face.right() - face.left()

        # 根据人脸大小生成空的图像
        img_blank = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            for j in range(width):
                img_blank[i][j] = img[face.top() + i][face.left() + j]

        # cv2.imshow("face_"+str(num+1), img_blank)

        # 存在本地
        print("Save into:", path_save + "img_face_" + str(num + 1) + ".jpg")
        cv2.imwrite(path_save + "img_face_" + str(num + 1) + ".jpg", img_blank)


if __name__ == '__main__':
    main()
