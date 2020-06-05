import cv2
import numpy as np
from handSeg import ycrcbSeg

def video_demo(cameraId):
    cam = cv2.VideoCapture(cameraId)
    while(True):
        ref, frame = cam.read()
        if(not ref):
            print("Wrong:cannot load camera!")
            exit(-1)
        frame = frame[:,::-1,:]
        cv2.imshow("test",frame)
        if((cv2.waitKey(30)&0xff)==27):
            cam.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    cameraId = 0        # 摄像头ID
    cam = cv2.VideoCapture(cameraId)

    # 强制视频格式转换，防止YUK格式帧率过低
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cam.set(cv2.CAP_PROP_FOURCC, fourcc)
    print('FrameWidth:', cam.get(cv2.CAP_PROP_FRAME_WIDTH), ', FrameHeight:', cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while(True):
        ref, frame = cam.read()
        if(not ref):
            print("Warn:cannot load camera!!!")
            exit(-1)
        frame = frame[:, ::-1, :]       # 翻转图像
        frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)      # 为合并窗口做准备
        _, crPic, ycrcbHand = ycrcbSeg(frame)       # ycrcb处理

        frameMerge = np.hstack((frameGray, ycrcbHand))      # 合并窗口
        cv2.imshow("Camera",frameMerge)
        # 加字未完成
        if ((cv2.waitKey(30) & 0xff) == 27):        # 按下ESC退出
            cam.release()
            cv2.destroyAllWindows()
            exit(0)