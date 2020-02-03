import cv2
import sys
import gc
from model import Model
 
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
        
    model = Model()
    model.load_model(file_path = '/home/luweijie/Videos/Face Detection/data/face_model.h5')
              
    color = (0, 255, 0)
    
    cap = cv2.VideoCapture(0)
    
    cascade_path = "/home/luweijie/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"
    
    while True:
        ret, frame = cap.read()
        
        if ret is True:
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        cascade = cv2.CascadeClassifier(cascade_path)

        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                
                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)
                
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'Oumar',(x + 30, y + 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                elif faceID == 1:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'Weijie',(x + 30, y + 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif faceID == 2:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'Junjie',(x + 30, y + 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                elif faceID == 3:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'Loic',(x + 30, y + 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("Face Preception", frame)
        
        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()