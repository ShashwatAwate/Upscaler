import tensorflow as tf
from keras import backend
import cv2
import os
import numpy as np

def perceptual_loss(y, y_hat):
    true_features = model(y)
    pred_features= model(y_hat)

    return backend.mean(backend.square(true_features - pred_features))


def combined_loss(y,y_hat):
    perceptual_loss = perceptual_loss(y,y_hat)
    pixel_loss = backend.mean(backend.square(y - y_hat))
    return 0.4*pixel_loss + perceptual_loss

def psnr(y, y_hat):
    return tf.image.psnr(y, y_hat, max_val=1.0)

model= tf.keras.models.load_model('./saved_model',
                                    custom_objects={'perceptual_loss':perceptual_loss , 'psnr':psnr,'combined_loss':combined_loss})


input_img = r'D:\coding\Upscaler\test\inp\0.png'
output_dir = './outputs'
os.makedirs(output_dir,exist_ok=True)

img = cv2.imread(input_img)
img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(256,256))
img = img/255.0
img = np.expand_dims(img,axis=0)



pred = model.predict(img)

print("Prediction shape:", pred.shape)
print("Min pixel value:", pred.min())
print("Max pixel value:", pred.max())

pred = pred*255.0
pred = np.clip(pred,0,255)
pred = np.squeeze(pred,axis=0)
pred = pred.astype(np.uint8)

output_image_path = os.path.join(output_dir, 'processed_image.jpg')
cv2.imwrite(output_image_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))