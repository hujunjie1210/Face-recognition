import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 64


def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    h, w, _ = image.shape
    
    longest_edge = max(h, w)
    
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    
    BLACK = [0, 0, 0]
    
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    return cv2.resize(constant, (height, width))

images = []
paths = []
labels = []
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                paths.append(path_name)
                    
    return images,paths
    
def load_dataset(path_name):
    images,paths = read_path(path_name)
    images = np.array(images)
    print(images.shape)
    
    #标注数据，'1'文件夹全部指定为0以此类推
    for path in paths:
        if path.endswith('1'):
            labels.append(0)
        if path.endswith('2'):
            labels.append(1)
        if path.endswith('3'):
            labels.append(2)
        if path.endswith('4'):
            labels.append(3)
    return images, labels

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        images, labels = load_dataset("/home/luweijie/Videos/Face Detection/data")
