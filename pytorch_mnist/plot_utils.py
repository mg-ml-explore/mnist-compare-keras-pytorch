import matplotlib.pyplot as plt

def plot_loss(train_loss, valid_loss):
    # Loss
    f = plt.figure(figsize=(10, 8))
    plt.plot(train_loss, label='training loss')
    plt.plot(valid_loss, label='validation loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_accuracy, valid_accuracy):
    # Accuracy
    f = plt.figure(figsize=(10, 8))
    plt.plot(train_accuracy, label='training accuracy')
    plt.plot(valid_accuracy, label='validation accuracy')
    plt.legend()
    plt.show()