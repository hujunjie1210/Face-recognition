import random
 
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
 
from load_dataset import load_dataset, resize_image, IMAGE_SIZE
 
 
 
class Dataset:
    def __init__(self, path_name):

        self.train_images = None
        self.train_labels = None
        
        self.valid_images = None
        self.valid_labels = None
        
        
        self.test_images  = None
        self.test_labels  = None
        
        
        self.path_name    = path_name
        
        
        self.input_shape = None
        
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, 
             img_channels = 3, nb_classes = 4):
        
        images, labels = load_dataset(self.path_name)
        
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5, random_state = random.randint(0, 100))
        
        print(train_labels)
        #channels,rows,cols，or:rows,cols,channels
        #first: th/last:tf
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
            
            
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')
        
            
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)

            print(train_labels)
            
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')
            
            
            train_images /= 255
            valid_images /= 255
            test_images /= 255
        
            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images  = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels  = test_labels
            
            
class Model:
    def __init__(self):
        self.model = None 
        
    
    def build_model(self, dataset, nb_classes = 4):
        
        self.model = Sequential() 
        
        
        self.model.add(Convolution2D(32, (3, 3), border_mode='same', 
                                     input_shape = dataset.input_shape))    #1
        self.model.add(Activation('relu'))                                  #2
        
        self.model.add(Convolution2D(32, (3, 3)))                           #3
        self.model.add(Activation('relu'))                                  #4
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #5
        self.model.add(Dropout(0.25))                                       #6

        self.model.add(Convolution2D(64, (3, 3), border_mode='same'))       #7
        self.model.add(Activation('relu'))                                  #8
        
        self.model.add(Convolution2D(64, (3, 3)))                           #9
        self.model.add(Activation('relu'))                                  #10
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #11
        self.model.add(Dropout(0.25))                                       #12
 
        self.model.add(Flatten())                                           #13
        self.model.add(Dense(512))                                          #14
        self.model.add(Activation('relu'))                                  #15
        self.model.add(Dropout(0.5))                                        #16
        self.model.add(Dense(nb_classes))                                   #17
        self.model.add(Activation('softmax'))                               #18
        
        
        self.model.summary()
        
    def train(self, dataset, batch_size = 20, nb_epoch = 4, data_augmentation = True):
        sgd = SGD(lr = 0.01, decay = 1e-6, 
                  momentum = 0.9, nesterov = True) #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        
        #不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        #训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:            
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           epochs = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
        #使用实时数据提升
        else:            
            #定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            #次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center = False,             #是否使输入数据去中心化（均值为0），
                samplewise_center  = False,             #是否使输入数据的每个样本均值为0
                featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
                zca_whitening = False,                  #是否对输入数据施以ZCA白化
                rotation_range = 20,                    #数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range  = 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range = 0.2,               #同上，只不过这里是垂直
                horizontal_flip = True,                 #是否进行随机水平翻转
                vertical_flip = False)                  #是否进行随机垂直翻转
 
            #计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)                        
 
            #利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     steps_per_epoch = dataset.train_images.shape[0],
                                     epochs = nb_epoch,
                                     verbose=1)    
    
    MODEL_PATH = '/home/luweijie/Videos/Face Detection/data/face_model.h5'
    def save_model(self, file_path = MODEL_PATH):
         self.model.save(file_path)
 
    def load_model(self, file_path = MODEL_PATH):
         self.model = load_model(file_path)
 
    def evaluate(self, dataset):
         score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
         print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def face_predict(self, image):
        
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        
        
        image = image.astype('float32')
        image /= 255
        
        
        result = self.model.predict_proba(image)
        print('result:', result)
        
        result = self.model.predict_classes(image)
 
        
        return result[0]

if __name__ == '__main__':
    dataset = Dataset('/home/luweijie/Videos/Face Detection/data')
    dataset.load()
    
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path = '/home/luweijie/Videos/Face Detection/data/face_model.h5')
    model.evaluate(dataset)