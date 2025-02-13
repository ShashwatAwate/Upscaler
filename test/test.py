import tensorflow as tf
from keras import backend,models
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

vgg = VGG19(include_top=False,weights='imagenet',input_shape=(None,None,3))
vgg.trainable=False
model = models.Model(inputs=vgg.input,outputs=vgg.get_layer('block3_conv3').output)

def perceptual_loss(y, y_hat):
    y = preprocess_input(y*255.0)
    y_hat = preprocess_input(y_hat*255.0)
    true_features = model(y)
    pred_features= model(y_hat)

    return backend.mean(backend.square(true_features - pred_features))


def combined_loss(y,y_hat):
    perceptual = perceptual_loss(y,y_hat)
    pixel_loss = backend.mean(backend.square(y - y_hat))
    l1_loss = backend.mean(backend.abs(y-y_hat))
    return 0.7*pixel_loss + 0.2*perceptual+0.2*l1_loss

def psnr(y, y_hat):
    return tf.image.psnr(y, y_hat, max_val=1.0)



model= tf.keras.models.load_model('./saved_model',
                                    custom_objects={'perceptual_loss':perceptual_loss , 'psnr':psnr,'combined_loss':combined_loss})


input_img = r'D:\coding\Upscaler\test\inp\0.png'
output_dir = './outputs'
os.makedirs(output_dir,exist_ok=True)
output_image_path = os.path.join(output_dir, 'processed_image.jpg')


img = cv2.imread(input_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32) / 255.0
img_batch = np.expand_dims(img, axis=0)

pred = model.predict(img_batch)

def compare_pred(input_img_path, model):
    # Load and preprocess image
    img = cv2.imread(input_img_path)
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(original_img, (128, 128))
    img = img.astype(np.float32) / 255.0
    

    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch)
    
    pred1 = np.squeeze(pred.copy(), axis=0)
    pred1 = np.clip(pred1,0.0,1.0)
    pred1 = (pred1 * 255).astype(np.uint8)
    

    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Input')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(pred1)
    plt.title('Upscaled')
    plt.axis('off')
    
    out = plt.imread(r"D:\coding\Upscaler\test\outp\0.png")
    plt.subplot(133)
    plt.imshow(out)
    plt.title("Actual Upscaled")
    plt.axis('off')
    plt.savefig(output_image_path)
    plt.close()
    
compare_pred(input_img_path=input_img,model=model)