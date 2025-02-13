import tensorflow as tf
from keras import models,backend
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,Activation,Add,UpSampling2D,Input,BatchNormalization
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
    l1_loss = backend.mean(backend.abs(y-y_hat))
    return 0.3*pixel_loss + 0.6*perceptual+0.01*l1_loss

def psnr(y,y_hat):
    return tf.image.psnr(y,y_hat,max_val=1.0)



def res_layer(x, skip, filter):
    if backend.int_shape(skip)[-1] != filter:
        skip = Conv2D(filter, (1,1), padding='same', activation=None)(skip)
    if backend.int_shape(x)[1] != backend.int_shape(skip)[1]:  
        skip = Conv2D(filter, (1,1), padding='same', activation=None)(skip)

    x = Conv2D(filter, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(filter, (3,3), padding='same', activation=None)(x)
    x = Add()([x, skip])
    x = Activation('relu')(x) 
    return x


def create_nn():
    r = 2
    inp = (Input(shape=(None,None,3)))
    x = Conv2D(64,(5,5),kernel_regularizer=l2(0.001),padding="same")(inp)
    x = Conv2D(32,(3,3),padding="same",kernel_regularizer=l2(0.001))(x)
    x = Conv2D(r**2*3,(3,3),padding="same",kernel_regularizer=l2(0.001))(x)
    outputs = tf.nn.depth_to_space(x,r)

    model = models.Model(inp,outputs)

    start_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(start_rate,first_decay_steps=40,t_mul=2.0,m_mul=0.9,alpha=1e-5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1.0)
    model.compile(optimizer=optimizer,loss = combined_loss ,metrics=[psnr])
    return model

model = create_nn()
model.summary()