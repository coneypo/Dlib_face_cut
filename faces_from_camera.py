# 调用摄像头，进行人脸捕获，和 68 个特征点的追踪

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_detection_from_camera

import dlib         # 人脸识别的库 Dlib
import cv2          # 图像处理的库 OpenCv
import time
import numpy as np

# 储存截图的目录
path_screenshots = "data/images/screenshots/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

# 创建 cv2 摄像头对象
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId 设置的视频参数，value 设置的参数值
cap.set(3, 960)

# 截图 screenshots 的计数器
cnt = 0

# cap.isOpened() 返回 true/false 检查初始化是否成功
while cap.isOpened():

    # cap.read()
    # 返回两个值：
    #    一个布尔值 true/false，用来判断读取视频是否成功/是否到视频末尾
    #    图像对象，图像的三维矩阵
    flag, img_rd = cap.read()

    # 每帧数据延时 1ms，延时为 0 读取的是静态帧
    k = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数
    faces = detector(img_gray, 0)

    # print(len(faces))

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 按下 'q' 键退出
    if k == ord('q'):
        break
    else:
        if len(faces) != 0:
            # 检测到人脸
            for kk, d in enumerate(faces):
                # 绘制矩形框
                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

                height = d.bottom() - d.top()
                width = d.right() - d.left()
                
                # 生成用来显示的图像
                img_blank = np.zeros((height, width, 3), np.uint8)

                # 记录每次开始写入人脸像素的宽度位置
                blank_start = 0

                # 将人脸填充到img_blank
                for k, d in enumerate(faces):

                    height = d.bottom() - d.top()
                    width = d.right() - d.left()

                    if blank_start + width >480:
                        break
                    else:
                        # 填充
                        for i in range(height):
                            for j in range(width):
                                img_rd[i][blank_start + j] = img_rd[d.top() + i][d.left() + j]
                        # 调整图像
                        blank_start += width

            cv2.putText(img_rd, "Faces in all: " + str(len(faces)), (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        else:
            # 没有检测到人脸
            cv2.putText(img_rd, "no face", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        # 添加说明
        img_rd = cv2.putText(img_rd, "Press 'S': Screen shot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        img_rd = cv2.putText(img_rd, "Press 'Q': Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 按下 's' 键保存
    if k == ord('s'):
        cnt += 1
        print(path_screenshots + "screenshot" + "_" + str(cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
        cv2.imwrite(path_screenshots + "screenshot" + "_" + str(cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", img_rd)

    cv2.namedWindow("camera", 1)
    cv2.imshow("camera", img_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()