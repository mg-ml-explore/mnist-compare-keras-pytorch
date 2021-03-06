from time import time

from . import data_utils
from . import model
from . import train_test_utils

def run(use_cuda: bool = False):
    device = "cpu"
    if use_cuda:
        device = "cuda"

    # Load train data
    mnist_train = data_utils.load_train_data()

    # Define model
    net = model.MNISTNet()
    net.to(device)
    net.summary(device)

    # Train model
    start = time()
    train_test_utils.train(net, mnist_train, device, use_cuda)
    end = time()
    print(f"Training duration: {end - start} seconds")

    # Load test data
    mnist_test = data_utils.load_test_data()
    # Test with trained model
