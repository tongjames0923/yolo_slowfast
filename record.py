
import cv2
import threading    #python 多线程操作库

class RecordingThread(threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True
        self.cap = camera
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') #设置视频编码方式
        self.out = cv2.VideoWriter(f'./videos/{name}.avi', fourcc, 20.0, (width, height))
        # out 是VideoWriter的实列对象，就是写入视频的方式，第一个参数是存放写入视频的位置，
        # 第二个是编码方式，20是帧率，最后是视频的高宽，如果录入视频为灰度，则还需加一个false

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()  #read()函数表示按帧读取视频，success，frame是read()的两个返回值，
            # ret是布尔值——如果读取帧是正确的则返回True，如果文件读取到结尾则返回False，Frame表示的是每一帧的图像，是一个三维矩阵
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()


class VideoCamera(object):
    def __init__(self):
        # 打开摄像头， 0代表笔记本内置摄像头
        self.cap = cv2.VideoCapture(0)
        self.cnt=0
        # 初始化视频录制环境
        self.is_record = False
        self.out = None

        # 视频录制线程
        self.recordingThread = None

    # 退出程序释放摄像头
    def __del__(self):
        self.cap.release()

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread(f"Video{self.cnt}", self.cap)
        self.cnt+=1
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()

import time
camera = VideoCamera()
while not (cv2.waitKey(33) & 0xFF == ord('q')):
 camera.start_record()
 time.sleep(2)
 camera.stop_record()