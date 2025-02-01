import tensorflow as tf
from keras import models,backend
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,Activation,Add,UpSampling2D,Input,BatchNormalization
from keras.regularizers import l2
vgg = VGG19(include_top=False,weights='imagenet',input_shape=(256,256,3))
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
    inp = (Input(shape=(256,256,3)))
    
    x = Conv2D(128,(3,3),padding="same",kernel_regularizer=l2(1e-7))(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)  #128 channels, 128x128
    skip1 = x                   #(128,128,128)
    print("block 1_maxpool1(128)",x.shape)

    x = Conv2D(256,(3,3),padding="same",kernel_regularizer=l2(1e-7))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip2 = x                  
    skip2 = MaxPooling2D((2,2))(skip2)   #(64,64,256)
    x = res_layer(x,skip1,256) #256 channels, 128x128
    print("block 2_maxpool_none(128)",x.shape)

    x = Conv2D(256,(3,3),padding="same",kernel_regularizer=l2(1e-7))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = res_layer(x,skip1,256)
    x = MaxPooling2D((2,2))(x) #256 channels, 64x64
    print("block 3_maxpool2(64)",x.shape)

    x = Conv2D(256,(3,3),padding="same",kernel_regularizer=l2(1e-7))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = res_layer(x,skip2,256) #256 channels, 64x64
    print("block 4_maxpool_none(64)",x.shape)

    x = Conv2DTranspose(256,(3,3),padding="same",kernel_regularizer=l2(1e-7))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x) #256 channels, 128x128
    print("block 5_upsample1(128)",x.shape)

    x = Conv2DTranspose(128,(3,3),padding="same",kernel_regularizer=l2(1e-7))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2,2))(x) # 128 channels, 256x256
    print("block 6_upsample2(256)",x.shape)

    x = Conv2DTranspose(128,(3,3),padding="same",kernel_regularizer=l2(1e-7))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip1 = UpSampling2D((2,2))(skip1)
    x = res_layer(x,skip1,128) # 128 channels, 256x256
    print("block 7_upsample_none(256)",x.shape)

    out= Conv2D(3,(3,3),activation='sigmoid',padding='same',kernel_regularizer=l2(1e-7))(x)
    print("final shape:",out.shape)
    model = models.Model(inputs=inp,outputs = out)

    start_rate = 1e-5
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(start_rate,first_decay_steps=40,t_mul=2.0,m_mul=0.9,alpha=1e-7)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1.0)
    model.compile(optimizer=optimizer,loss = combined_loss ,metrics=[psnr])
    return model

model = create_nn()
model.summary()