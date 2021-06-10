# 调用摄像头实时单个/多个人脸检测，并依次在摄像头窗口，实时平铺显示检测到的人脸；

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_cut

import dlib
import cv2
import time
import os


def main():
    # 储存截图的目录
    path_screenshots = "data/images/screenshots/"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

    # 创建 cv2 摄像头对象
    cap = cv2.VideoCapture(0)

    # 设置视频参数，propId 设置的视频参数，value 设置的参数值
    cap.set(3, 960)

    # 截图 screenshots 的计数器
    ss_cnt = 0

    while cap.isOpened():
        flag, img_rd = cap.read()

        # 每帧数据延时 1ms，延时为 0 读取的是静态帧
        k = cv2.waitKey(1)

        # 取灰度
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

        # 人脸数
        faces = detector(img_gray, 0)

        # 待会要写的字体
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 按下 'q' 键退出
        if k == ord('q'):
            break
        else:
            # 检测到人脸
            if len(faces) != 0:
                # 记录每次开始写入人脸像素的宽度位置
                faces_start_width = 0

                for face in faces:

                    # 人脸对齐 / get face chip
                    face_shape = predictor(img_rd, face)
                    face_chip = dlib.get_face_chip(img_rd, face_shape)

                    # 绘制矩形框
                    cv2.rectangle(img_rd, tuple([face.left(), face.top()]), tuple([face.right(), face.bottom()]),
                                  (0, 255, 255), 2)

                    # 如果没有超出摄像头边界
                    for i in range(150):
                        for j in range(150):
                            img_rd[i][faces_start_width + j] = \
                                face_chip[i][j]

                    # 更新 faces_start_width 的坐标
                    faces_start_width += 150

                cv2.putText(img_rd, "Faces in all: " + str(len(faces)), (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            else:
                # 没有检测到人脸
                cv2.putText(img_rd, "no face", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            # 添加说明
            img_rd = cv2.putText(img_rd, "Press 'S': Screen shot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            img_rd = cv2.putText(img_rd, "Press 'Q': Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # 按下 's' 键保存
        if k == ord('s'):
            if not os.path.isdir(path_screenshots):
                os.mkdir(path_screenshots)
            ss_cnt += 1
            print(path_screenshots + "screenshot" + "_" + str(ss_cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                            time.localtime()) + ".jpg")
            cv2.imwrite(path_screenshots + "screenshot" + "_" + str(ss_cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                                  time.localtime()) + ".jpg",
                        img_rd)

        cv2.namedWindow("camera", 1)
        cv2.imshow("camera", img_rd)

    # 释放摄像头
    cap.release()

    # 删除建立的窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
