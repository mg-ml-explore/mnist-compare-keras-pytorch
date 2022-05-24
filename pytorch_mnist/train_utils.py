import torch
from torch import cuda
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torch import nn
from . import data_utils

def train(
    net: nn.Module,
    train_data: MNIST,
    valid_data: MNIST,
    use_cuda,
    num_epochs=10,
    batch_size=1024,
):
    mnist_train_loader = data_utils.get_data_loader(train_data, batch_size, use_cuda)
    mnist_valid_loader = data_utils.get_data_loader(valid_data, batch_size, use_cuda)

    if cuda.is_available():
        net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) 

    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    
    for epoch in range(num_epochs):
        
        ############################
        # Train
        ############################

        iter_loss = 0.0
        correct = 0
        iterations = 0
        
        net.train()                   # Put the network into training mode
        
        for i, (items, classes) in enumerate(mnist_train_loader):
            
            # Convert torch tensor to Variable
            items = Variable(items)
            classes = Variable(classes)
            
            # If we have GPU, shift the data to GPU
            if cuda.is_available():
                items = items.cuda()
                classes = classes.cuda()
            
            optimizer.zero_grad()     # Clear off the gradients from any past operation
            outputs = net(items)      # Do the forward pass
            loss = criterion(outputs, classes) # Calculate the loss
            iter_loss += loss.data    # Accumulate the loss
            loss.backward()           # Calculate the gradients with help of back propagation
            optimizer.step()          # Ask the optimizer to adjust the parameters based on the gradients
            
            # Record the correct predictions for training data 
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data).sum()
            iterations += 1
        
        # Record the training loss
        train_loss.append(iter_loss/iterations)
        # Record the training accuracy
        train_accuracy.append((100 * correct / len(mnist_train_loader.dataset)))
    

        ############################
        # Validate - How did we do on the unseen dataset?
        ############################
        
        loss = 0.0
        correct = 0
        iterations = 0

        net.eval()                    # Put the network into evaluate mode
        
        for i, (items, classes) in enumerate(mnist_valid_loader):
            
            # Convert torch tensor to Variable
            items = Variable(items)
            classes = Variable(classes)
            
            # If we have GPU, shift the data to GPU
            if cuda.is_available():
                items = items.cuda()
                classes = classes.cuda()
            
            outputs = net(items)      # Do the forward pass
            loss += criterion(outputs, classes).data # Calculate the loss
            
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data).sum()
            
            iterations += 1

        # Record the validation loss
        valid_loss.append(loss/iterations)
        # Record the validation accuracy
        valid_accuracy.append(correct / len(mnist_valid_loader.dataset) * 100.0)

        
        print ('Epoch %d/%d, Tr Loss: %.4f, Tr Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f'
            %(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], 
                valid_loss[-1], valid_accuracy[-1]))

    return {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
    }
