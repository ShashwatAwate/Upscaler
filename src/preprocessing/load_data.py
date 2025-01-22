import cv2
import numpy
import os
import tensorflow as tf
import numpy as np

data_path = './data/'
train_path =  os.path.join(data_path,'train/')
val_path = os.path.join(data_path,'val/')

train_high = r'D:\coding\Upscaler\data\\train\high_res'
train_low = r'D:\coding\Upscaler\data\\train\low_res'

val_high = r'D:\coding\Upscaler\data\\val\high_res'
val_low = r'D:\coding\Upscaler\data\\val\low_res'

print('creating a tf record')
with tf.io.TFRecordWriter('data.tfrecord') as tfrecord:
    for filename in os.listdir(train_low):

        low_res_path = os.path.join(train_low,filename)
        high_res_path = os.path.join(train_high,filename)

        if os.path.exists(high_res_path):
            low_res_img = cv2.imread(low_res_path)
            high_res_img = cv2.imread(high_res_path)

            low_res_img = cv2.cvtColor(low_res_img,cv2.COLOR_BGR2RGB)
            high_res_img = cv2.cvtColor(high_res_img,cv2.COLOR_BGR2RGB)

            target_size = (256,256)
            low_res_img = cv2.resize(low_res_img,target_size)
            high_res_img = cv2.resize(high_res_img,target_size)

            low_res_img = low_res_img/255.0
            high_res_img = high_res_img/255.0

            low_res_img = (low_res_img*255).astype(np.float32)
            high_res_img = (high_res_img*255).astype(np.float32)
            
            # low_res_img = np.expand_dims(low_res_img,axis=0)
            # high_res_img = np.expand_dims(high_res_img,axis=0)

            def create_feature(value):
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
            
            low_res_bytes = low_res_img.tobytes()
            high_res_bytes = high_res_img.tobytes()

            feature = {
                'low_res': create_feature(low_res_bytes),
                'high_res': create_feature(high_res_bytes),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            tfrecord.write(example.SerializeToString())

with tf.io.TFRecordWriter('data_val.tfrecord') as tfrecord:
    for filename in os.listdir(val_low):

        low_res_path = os.path.join(val_low,filename)
        high_res_path = os.path.join(val_high,filename)

        if os.path.exists(high_res_path):
            low_res_img = cv2.imread(low_res_path)
            high_res_img = cv2.imread(high_res_path)

            low_res_img = cv2.cvtColor(low_res_img,cv2.COLOR_BGR2RGB)
            high_res_img = cv2.cvtColor(high_res_img,cv2.COLOR_BGR2RGB)

            target_size = (256,256)
            low_res_img = cv2.resize(low_res_img,target_size)
            high_res_img = cv2.resize(high_res_img,target_size)

            low_res_img = low_res_img/255.0
            high_res_img = high_res_img/255.0

            low_res_img = (low_res_img*255).astype(np.float32)
            high_res_img = (high_res_img*255).astype(np.float32)

            def create_feature(value):
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
            
            low_res_bytes = low_res_img.tobytes()
            high_res_bytes = high_res_img.tobytes()

            feature = {
                'low_res': create_feature(low_res_bytes),
                'high_res': create_feature(high_res_bytes),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            tfrecord.write(example.SerializeToString())

print('tf records created!!!')






