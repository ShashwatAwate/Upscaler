import cv2
import numpy
import os
import tensorflow as tf
import numpy as np

data_path = './data/'
train_path =  os.path.join(data_path,'train/')
val_path = os.path.join(data_path,'val/')

def create_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def scale(img,scale):
    h,w,ch = img.shape
    nh,nw = int(h*scale),int(w*scale)
    if scale <1:
        img = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_CUBIC)
    
    return cv2.resize(img,(w,h),interpolation=cv2.INTER_CUBIC)

def rotate(img,angle):
    h,w,ch = img.shape
    mat = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    return cv2.warpAffine(img,mat,(w,h),borderMode=cv2.BORDER_REFLECT)

def augmentation(low_img,high_img):
    if np.random.rand()<0.5:
        angle =np.random.uniform(-10,10) 
        low_img = rotate(low_img,angle)
        high_img = rotate(high_img,angle)

    if np.random.rand()<0.5:
        scale_factor = np.random.uniform(0.8,1.2)
        low_img = scale(low_img,scale_factor)
        high_img = scale(high_img,scale_factor)

    if np.random.rand()<0.3:
        kernel = np.random.choice([3,5])
        low_img = cv2.GaussianBlur(low_img,(kernel,kernel),sigmaX=1.0)
        high_img = cv2.GaussianBlur(high_img,(kernel,kernel),sigmaX=1.0)

    if np.random.rand()<0.3:
        noise = np.random.normal(0,0.02,low_img.shape).astype(np.float32)
        low_img = np.clip(low_img+noise,0,1)
        high_img = np.clip(high_img+noise,0,1)

    return low_img,high_img

def write_TFRecord(low_path,high_path,name):

    with tf.io.TFRecordWriter(name) as writer:

        for filename in os.listdir(low_path):

            low_res_path = os.path.join(low_path,filename)
            high_res_path = os.path.join(high_path,filename)

            if os.path.exists(high_res_path) and os.path.exists(low_res_path):
                low_res_img = cv2.imread(low_res_path)
                high_res_img = cv2.imread(high_res_path)

                low_res_img = cv2.cvtColor(low_res_img,cv2.COLOR_BGR2RGB)
                high_res_img = cv2.cvtColor(high_res_img,cv2.COLOR_BGR2RGB)

                target_size = (256,256)
                low_res_img = cv2.resize(low_res_img,target_size)
                high_res_img = cv2.resize(high_res_img,target_size)

                # low_res_img = low_res_img/255.0
                # high_res_img = high_res_img/255.0

                # low_res_img = (low_res_img*255).astype(np.float32)
                # high_res_img = (high_res_img*255).astype(np.float32)

                low_res_img  = low_res_img.astype(np.float32)
                high_res_img  = high_res_img.astype(np.float32)

                low_res_original = low_res_img
                high_res_original = high_res_img

                low_res_bytes = low_res_img.tobytes()
                high_res_bytes = high_res_img.tobytes()

                feature = {
                    'low_res': create_feature(low_res_bytes),
                    'high_res': create_feature(high_res_bytes),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

                for _ in range(10):
                    aug_low,aug_high = augmentation(low_res_original,high_res_original)
                    aug_low_bytes = aug_low.tobytes()
                    aug_high_bytes = aug_high.tobytes()
                    f = {
                        'low_res':create_feature(aug_low_bytes),
                        'high_res':create_feature(aug_high_bytes),
                    }
                    ex = tf.train.Example(features=tf.train.Features(feature=f))
                    writer.write(ex.SerializeToString())

train_high = r'D:\coding\Upscaler\data\\train\high_res'
train_low = r'D:\coding\Upscaler\data\\train\low_res'

val_high = r'D:\coding\Upscaler\data\\val\high_res'
val_low = r'D:\coding\Upscaler\data\\val\low_res'

val_record_path = r'D:\coding\Upscaler\data\data_val.tfrecord'
train_record_path = r'D:\coding\Upscaler\data\data.tfrecord'

print("Writing Train record....")
write_TFRecord(low_path=train_low,high_path=train_high,name=train_record_path)
print("Train record created !!!!")
print("Writing val record....")
write_TFRecord(low_path=val_low,high_path=val_high,name=val_record_path)
print("Val record created !!!!")

print('tf records created!!!')






