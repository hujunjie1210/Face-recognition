import cv2
import sys
from PIL import Image
 
def CatchUsbVideo(catch_pic_num, path_name):
    cv2.namedWindow("detect")
    
    cap = cv2.VideoCapture("1.mp4")
    
    classfier = cv2.CascadeClassifier("/home/luweijie/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
    
    #RGB
    color = (0, 255, 0)
    num=0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                img_name = '%s/%d.jpg'%(path_name, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)
                num += 1
                if num > (catch_pic_num):
                    break
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)
        
        if num > (catch_pic_num): break

        cv2.imshow("detect", frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo(300,'/home/luweijie/Videos/Face Detection/data/1')