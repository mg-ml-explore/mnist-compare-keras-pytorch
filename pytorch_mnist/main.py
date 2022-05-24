
from . import data_utils
from . import model
from . import train_test_utils

def run(use_cuda: bool):
    device = "cpu"
    if use_cuda:
        device = "cuda"

    # Load train data
    mnist_train = data_utils.load_train_data()

    # Define model
    net = model.MNISTNet()
    net.summary()
    net.to(device)

    # Train model
    output = train_test_utils.train(net, mnist_train, device, use_cuda)

    # Load test data
    mnist_test = data_utils.load_test_data()
    # Test with trained model
