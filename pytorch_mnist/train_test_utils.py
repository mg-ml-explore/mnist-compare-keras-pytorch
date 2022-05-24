import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import MNIST
from . import data_utils


def train_single_epoch(model: nn.Module, train_loader: DataLoader, optimizer, epoch, device: str):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model: nn.Module, test_loader: DataLoader, device: str):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train(model: nn.Module, train_valid_data: MNIST, device, use_cuda, batch_size = 1000, epochs = 10):
    train_data, valid_data = data_utils.split_train_valid(train_valid_data)

    train_loader = data_utils.get_data_loader(train_data, batch_size, use_cuda)
    valid_loader = data_utils.get_data_loader(valid_data, batch_size, use_cuda)

    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, epochs):
        train_single_epoch(model, train_loader, optimizer, epoch, device)
        test(model, valid_loader, device)

    # torch.save(model.state_dict(), "mnist_cnn.pt")
