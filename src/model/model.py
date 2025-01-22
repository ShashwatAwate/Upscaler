import tensorflow as tf
from keras import layers,models,backend

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

    x = (layers.Conv2D(64,(3,3),activation='relu',padding = 'same'))(inputs)
    x = (layers.MaxPooling2D((2,2)))(x)

    x = res_layer(x,64)

    x = (layers.Conv2D(128,(3,3),activation='relu',padding = 'same'))(x)
    x = (layers.MaxPooling2D((2,2)))(x)

    x = (res_layer(x,128))
    
    x = (layers.Conv2DTranspose(128,(3,3),activation='relu',padding = 'same'))(x)
    x = (layers.UpSampling2D((2,2)))(x)

    x = (res_layer(x,128))

    x = (layers.Conv2DTranspose(64,(3,3),activation='relu',padding = 'same'))(x)
    x = (layers.UpSampling2D((2,2)))(x)

    x = (res_layer(x,64))

    outputs = layers.Conv2D(3,(3,3),activation='sigmoid',padding='same')(x)
    model = models.Model(inputs=inputs,outputs = outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss = 'mean_squared_error',metrics=['accuracy'])
    return model

model = create_nn()
model.summary()