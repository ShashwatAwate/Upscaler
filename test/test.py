import tensorflow as tf
from keras import backend
import cv2
import os

def perceptual_loss(y, y_hat):
    true_features = model(y)
    pred_features= model(y_hat)

    return backend.mean(backend.square(true_features - pred_features))

def psnr(y, y_hat):
    return tf.image.psnr(y, y_hat, max_val=1.0)

model= tf.keras.models.load_model('./saved_model',
                                    custom_objects={'perceptual_loss':perceptual_loss , 'psnr':psnr})


input_img = r'D:\coding\Upscaler\test\inp\0.png'
output_dir = './outputs'
os.makedirs(output_dir,exist_ok=True)

img = cv2.imread(input_img)
img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img = tf.image.resize(img,(256,256))
img = img/255.0
img = tf.expand_dims(img,axis=0)
pred = model.predict(img)

pred = pred*255.0
output_image_path = os.path.join(output_dir, 'processed_image.jpg')
cv2.imwrite(output_image_path, cv2.cvtColor(pred[0].astype('uint8'), cv2.COLOR_RGB2BGR))