import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import MNIST
from . import data_utils


def train_single_epoch(model: nn.Module, train_loader: DataLoader, optimizer, epoch, device: str):
    model.train()

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).sum().item()

    print('\Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, train_correct, len(train_loader.dataset),
        100. * train_correct / len(train_loader.dataset)))

def test(model: nn.Module, test_loader: DataLoader, device: str):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train(model: nn.Module, train_valid_data: MNIST, device, use_cuda, batch_size = 128, epochs = 15):
    train_data, valid_data = data_utils.split_train_valid(train_valid_data)

    train_loader = data_utils.get_data_loader(train_data, batch_size, use_cuda)
    valid_loader = data_utils.get_data_loader(valid_data, batch_size, use_cuda)

    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, epochs):
        train_single_epoch(model, train_loader, optimizer, epoch, device)
        test(model, valid_loader, device)

    # torch.save(model.state_dict(), "mnist_cnn.pt")
