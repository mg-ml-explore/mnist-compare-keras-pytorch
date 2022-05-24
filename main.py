# from keras_mnist import main as keras_mnist
from pytorch_mnist import main as pytorch_mnist

# keras_mnist.run()
pytorch_mnist.run(use_cuda=False)
