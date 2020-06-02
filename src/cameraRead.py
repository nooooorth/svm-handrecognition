import cv2
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
    while(True):
        ref, frame = cam.read()
        if(not ref):
            print("Warn:cannot load camera!!!")
            exit(-1)
        frame = frame[:, ::-1, :]

        _, crPic, ycrcbHand = ycrcbSeg(frame)

        cv2.imshow("ycrcbHand", ycrcbHand)
        cv2.imshow("BGR", frame)
        if ((cv2.waitKey(30) & 0xff) == 27):
            cam.release()
            cv2.destroyAllWindows()
            exit(0)


