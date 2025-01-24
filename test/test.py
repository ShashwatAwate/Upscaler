import tensorflow as tf
import cv2
import os
model= tf.keras.models.load_model('./saved_model')


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