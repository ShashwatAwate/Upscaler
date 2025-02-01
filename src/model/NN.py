import tensorflow as tf
from data_ingestion import parse
from model import create_nn
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import mixed_precision

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def plot_data(history):
    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],color='blue',label='train_loss')
    plt.plot(history.history['val_loss'],color = 'orange', label = 'val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['psnr'],color='green',label='psnr')
    plt.plot(history.history['val_psnr'],color='red',label='val_psnr')
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


data_path = r'D:\coding\Upscaler\data\data.tfrecord'
val_path = r'D:\coding\Upscaler\data\data.tfrecord'

raw_dataset = tf.data.TFRecordDataset(data_path)
val_raw_dataset = tf.data.TFRecordDataset(val_path)

batch_size = 8

raw_dataset = tf.data.TFRecordDataset(data_path)
parsed_dataset = raw_dataset.map(parse)
parsed_dataset = parsed_dataset.shuffle(1000).batch(16).repeat()

val_raw_dataset = tf.data.TFRecordDataset(val_path)
parsed_val_dataset = val_raw_dataset.map(parse)
parsed_val_dataset = parsed_val_dataset.shuffle(1000).batch(16).repeat()
# print('train examples',sum(1 for _ in parsed_dataset))
# print('test examples',sum(1 for _ in parsed_val_dataset))
save_path = r'D:\coding\Upscaler\test\saved_model'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
]
model = create_nn()
history = model.fit(
    parsed_dataset,
    validation_data=parsed_val_dataset,
    batch_size=batch_size,
    epochs=100,
    steps_per_epoch=925,
    validation_steps=50,
    callbacks=callbacks
)
model.save(r'D:\coding\Upscaler\\test\saved_model')
plot_data(history)

