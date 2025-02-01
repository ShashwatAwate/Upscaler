import tensorflow as tf
from keras import backend,models
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

vgg = VGG19(include_top=False,weights='imagenet',input_shape=(256,256,3))
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

def debug_psnr(y_true, y_pred):
    # Original PSNR calculation
    tf_psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
    
    # Manual PSNR calculation for verification
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:  # If images are identical
        manual_psnr = 100
    else:
        manual_psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Print detailed statistics
    print("PSNR Debug Information:")
    print(f"TF PSNR: {tf_psnr.numpy()}")
    print(f"Manual PSNR: {manual_psnr}")
    print(f"MSE: {mse}")
    print(f"Input range: [{y_true.min()}, {y_true.max()}]")
    print(f"Prediction range: [{y_pred.min()}, {y_pred.max()}]")
    
    # Check for numerical issues
    print("\nNumerical Checks:")
    print(f"Contains NaN - Input: {np.any(np.isnan(y_true))}, Pred: {np.any(np.isnan(y_pred))}")
    print(f"Contains Inf - Input: {np.any(np.isinf(y_true))}, Pred: {np.any(np.isinf(y_pred))}")
    
    return manual_psnr

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

# Calculate PSNR with debugging
debug_psnr(img_batch, pred) 

# img = cv2.imread(input_img)
# img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = cv2.resize(img,(256,256))
# img = img/255.0
# print("Input image range:", img.min(), img.max())  # Add this
# img = np.expand_dims(img,axis=0)
# pred = model.predict(img)
# print("Raw prediction range:", pred.min(), pred.max())
# pred_numpy = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
# pred_numpy = np.squeeze(pred_numpy, axis=0)
# plt.imsave(output_image_path, pred_numpy)  # Remove the cv2.cvtColor step

def debug_prediction(input_img_path, model):
    # Load and preprocess image
    img = cv2.imread(input_img_path)
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(original_img, (256, 256))
    img = img.astype(np.float32) / 255.0
    
    # Make prediction
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch)
    
    # Try different post-processing approaches
    
    # Approach 1: Direct scaling
    pred1 = np.squeeze(pred.copy(), axis=0)
    pred1 = (pred1 * 255).astype(np.uint8)
    
    # Approach 2: Channel-wise normalization
    pred2 = np.squeeze(pred.copy(), axis=0)
    for c in range(3):
        channel = pred2[:,:,c]
        if channel.max() != channel.min():
            pred2[:,:,c] = (channel - channel.min()) / (channel.max() - channel.min())
    pred2 = (pred2 * 255).astype(np.uint8)
    
    # Approach 3: Gamma correction
    pred3 = np.squeeze(pred.copy(), axis=0)
    gamma = 0.8  # Adjust this value if needed
    pred3 = np.power(pred3, gamma)
    pred3 = (pred3 * 255).astype(np.uint8)
    
    # Visualize all approaches
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(img)
    plt.title('Input')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(pred1)
    plt.title('Direct Scaling')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(pred2)
    plt.title('Channel Normalized')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(pred3)
    plt.title('Gamma Corrected')
    plt.axis('off')
    
    plt.savefig('debug_approaches.png')
    plt.close()
    
    # Print detailed statistics
    print("\nPrediction Statistics:")
    print(f"Raw prediction shape: {pred.shape}")
    print(f"Raw prediction range: [{pred.min():.6f}, {pred.max():.6f}]")
    print("\nChannel-wise statistics:")
    for i in range(3):
        channel = pred[0,:,:,i]
        print(f"Channel {i}:")
        print(f"  Range: [{channel.min():.6f}, {channel.max():.6f}]")
        print(f"  Mean: {channel.mean():.6f}")
        print(f"  Std: {channel.std():.6f}")
        unique_values = len(np.unique(channel))
        print(f"  Unique values: {unique_values}")

    return pred1, pred2, pred3

# Try the debugging function
pred1, pred2, pred3 = debug_prediction(input_img, model)