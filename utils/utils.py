import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history['loss'],color='blue')
    plt.plot(history['val_loss'],color = 'orange')
    plt.grid(True)
    plt.show()

