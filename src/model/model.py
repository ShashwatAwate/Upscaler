import tensorflow as tf
from keras import layers,models,backend
from keras.applications import VGG19


vgg = VGG19(include_top=False,weights='imagenet',input_shape=(256,256,3))
vgg.trainable=False
model = models.Model(inputs=vgg.input,outputs=vgg.get_layer('block5_conv4').output)


def perceptual_loss(y,y_hat):
    
    true_features = model(y)
    pred_features= model(y_hat)

    return backend.mean(backend.square(true_features - pred_features))

def psnr(y,y_hat):
    return tf.image.psnr(y,y_hat,max_val=1.0)


def res_layer(x,filter):
    inp_sig = x
    if backend.int_shape(inp_sig)[-1]!=filter:
        inp_sig = layers.Conv2D(filter,(1,1),activation=None,padding='same')(inp_sig)
    x = layers.Conv2D(filter,(3,3),activation='relu',padding='same')(x)
    x = layers.Conv2D(filter,(3,3),activation=None,padding='same')(x)
    x = layers.Add()([x,inp_sig])
    x = layers.Activation('relu')(x)
    return x

def create_nn():
    inputs = (layers.Input(shape=(256,256,3)))

    x = (layers.Conv2D(64,(3,3),padding = 'same'))(inputs) 
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = (layers.MaxPooling2D((2,2)))(x) #128
    print(f"Shape after downsample block 1 {x.shape}")
    x = res_layer(x,64)

    x = (layers.Conv2D(128,(3,3),padding = 'same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = (layers.MaxPooling2D((2,2)))(x) #64
    print(f"Shape after downsample block 2 {x.shape}")
    x = (res_layer(x,128))

    x = (layers.Conv2D(128,(3,3),padding = 'same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = (layers.MaxPooling2D((2,2)))(x)  #32
    print(f"Shape after downsample block 3 {x.shape}")
    x = (res_layer(x,128))

    x = (layers.Conv2DTranspose(128,(3,3),padding = 'same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = (layers.UpSampling2D((2,2)))(x)  #64
    x = (res_layer(x,128))
    print(f"Shape after upsample block 1 {x.shape}")

    x = (layers.Conv2DTranspose(128,(3,3),padding = 'same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = (layers.UpSampling2D((2,2)))(x)  #128
    x = (res_layer(x,128))
    print(f"Shape after upsample block 2 {x.shape}")
    

    x = (layers.Conv2DTranspose(128,(3,3),padding = 'same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = (layers.UpSampling2D((2,2)))(x)  #256
    x = (res_layer(x,128))
    print(f"Shape after upsample block 3 {x.shape}")

    outputs = layers.Conv2D(3,(3,3),activation='tanh',padding='same')(x)
    print("After final Conv2D (3 filters):", outputs.shape)
    
    model = models.Model(inputs=inputs,outputs = outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss = perceptual_loss ,metrics=[psnr])
    return model

model = create_nn()
model.summary()