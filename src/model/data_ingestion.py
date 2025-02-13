import tensorflow as tf
import numpy as np


def parse(proto):
    feature_des = {
    'low_res' : tf.io.FixedLenFeature([],tf.string),
    'high_res' : tf.io.FixedLenFeature([],tf.string)
}
    parsed_example = tf.io.parse_single_example(proto,feature_des)

    low_res = tf.io.decode_raw(parsed_example['low_res'],tf.float32)
    high_res = tf.io.decode_raw(parsed_example['high_res'],tf.float32)

    low_res = tf.reshape(low_res, [128, 128, 3]) 
    high_res = tf.reshape(high_res, [256, 256, 3]) 

    low_res = low_res/255.0 
    high_res = high_res/255.0

    low_res = tf.cast(low_res,tf.float32)
    high_res = tf.cast(high_res,tf.float32)
    
    low_res = tf.expand_dims(low_res,axis=0)
    high_res = tf.expand_dims(high_res,axis=0)

    return low_res,high_res

