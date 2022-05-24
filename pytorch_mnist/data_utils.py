from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

# Mean and standard deviation of all the pixels in the MNIST dataset
mean_gray = 0.1307
stddev_gray = 0.3081
transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((mean_gray,), (stddev_gray,))])
train_data = MNIST('./data', train=True, download=True, transform=transform)
test_data = MNIST('./data', train=False, download=True, transform=transform)
print("x_train shape:", train_data.data.size())
print(train_data.data.size()[0], "train samples")
print(test_data.data.size()[0], "test samples")

def load_train_data():
    return train_data

def load_test_data():
    return test_data

def split_train_valid(mnist_data: MNIST):
    train_size = int(0.9 * len(mnist_data))
    test_size = len(mnist_data) - train_size
    train_mnist_data, valid_mnist_data = random_split(mnist_data, [train_size, test_size])
    return train_mnist_data, valid_mnist_data

def get_data_loader(mnist_data: MNIST, batch_size: int, use_cuda: bool):
    data_loader_options = {
        "batch_size": batch_size,
        "shuffle": True,
        "pin_memory": True
    }
    if use_cuda:
        data_loader_options["num_workers"] = 1
    return DataLoader(mnist_data, **data_loader_options)
