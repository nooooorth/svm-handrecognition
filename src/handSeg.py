import cv2

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
    video_demo(cameraId)


