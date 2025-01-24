import tensorflow as tf
from data_ingestion import parse
from model import create_nn
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'],color='blue')
    plt.plot(history.history['val_loss'],color = 'orange')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.show()


data_path = r'D:\coding\Upscaler\data\data.tfrecord'
val_path = r'D:\coding\Upscaler\data\data.tfrecord'

raw_dataset = tf.data.TFRecordDataset(data_path)
parsed_dataset = raw_dataset.map(parse)

val_raw_dataset = tf.data.TFRecordDataset(val_path)
parsed_val_dataset = val_raw_dataset.map(parse)

# print('train examples',sum(1 for _ in parsed_dataset))
# print('test examples',sum(1 for _ in parsed_val_dataset))

batch_size = 10

model = create_nn()
history = model.fit(
    parsed_dataset,
    validation_data=parsed_val_dataset,
    epochs=10,
    steps_per_epoch=50,
    validation_steps=10
)
model.save(r'D:\coding\Upscaler\\test\saved_model')
plot_loss(history)

