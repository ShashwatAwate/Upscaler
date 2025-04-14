import tensorflow as tf
from keras import models,backend
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.layers import Conv2D,Add,Input,PReLU,Lambda
from keras.regularizers import l2
vgg = VGG19(include_top=False,weights='imagenet',input_shape=(None,None,3))
vgg.trainable=False
model = models.Model(inputs=vgg.input,outputs=vgg.get_layer('block3_conv3').output)


def perceptual_loss(y,y_hat):
    y = preprocess_input(y*255.0)
    y_hat = preprocess_input(y_hat*255.0)
    true_features = model(y)
    pred_features= model(y_hat)

    return backend.mean(backend.square(true_features - pred_features))

def combined_loss(y,y_hat):
    perceptual = perceptual_loss(y,y_hat)
    pixel_loss = backend.mean(backend.square(y - y_hat))
    total = 0.8*pixel_loss + 0.2*perceptual/tf.cast(tf.size(perceptual),tf.float32) 
    return total

def psnr(y,y_hat):
    return tf.image.psnr(y,y_hat,max_val=1.0)


def create_nn():
    r = 2
    inp = (Input(shape=(None,None,3)))
    x = Conv2D(56,(5,5),kernel_regularizer=l2(0.001),padding="same")(inp)
    x = PReLU(shared_axes=[1,2])(x)
    x = Conv2D(56,(1,1),padding="same",kernel_regularizer=l2(0.001))(x)
    x = PReLU(shared_axes=[1,2])(x)

    for _ in range(2):
        skip = x
        x = Conv2D(64,(3,3),padding="same",kernel_regularizer=l2(0.001))(x)
        x = PReLU(shared_axes=[1,2])(x)
        x = Conv2D(64,(3,3),padding="same",kernel_regularizer=l2(0.001))(x)
        x = PReLU(shared_axes=[1,2])(x)
        x = Conv2D(64,(3,3),padding="same",kernel_regularizer=l2(0.001))(x)
        x = PReLU(shared_axes=[1,2])(x)
        x = Conv2D(64,(3,3),padding="same",kernel_regularizer=l2(0.001))(x)
        x = PReLU(shared_axes=[1,2])(x)
        if x.shape[-1] != skip.shape[-1]:
            skip = Conv2D(x.shape[-1],(1,1),padding="same")(skip)
        x = Add()([x,skip])
    

    
    x = Conv2D(r**2*3,(3,3),padding="same",kernel_regularizer=l2(0.001))(x)
    outputs = Lambda(lambda x:tf.nn.depth_to_space(x,r))(x)
    model = models.Model(inp,outputs)

    start_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=start_rate,clipnorm=1.0)
    model.compile(optimizer=optimizer,loss = combined_loss ,metrics=[psnr])
    return model


if __name__=='__main__':
    
    model = create_nn()
    model.summary()