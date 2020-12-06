def eval(net, data_loader, device):
    """Returns accuracy of the passed network on the given dataset

    Parameters
    ----------
    net: nn.Module
        Neural network
    data_loader: torch.utils.data.DataLoader
        Loader that should be iterable and return batches of data
    device
        Device to be used for evaluating the network
    """
    net.to(device)
    net.eval()
    correctly_predicted = 0
    dataset_size = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        correctly_predicted += (net(x).argmax(axis=1) == y).float().sum()
        dataset_size += y.size(0)
    accuracy = correctly_predicted / float(dataset_size)
    return accuracy    

def train(net, data_loader, num_epochs, optimizer, loss_fn, device, scheduler=None):
    """Optimize neural network under criterion given by loss_fn

    Parameters
    ----------
    net: nn.Module
        Neural network
    data_loader: torch.utils.data.DataLoader
        Loader that should be iterable and return batches of data for the net to be trained on
    num_epochs: int
        Number of epochs
    optimizer
        Optimizer to be used for training
    loss_fn
        Criterion under which the net should be optimizer
    device
        Device to be used for training the network
    scheduler
        Learning rate scheduler
    """
    net.to(device)
    for i in range(num_epochs):
        print(f'Epoch {i}', end='')
        net.train()
        loss_accumulator = 0
        dataset_size = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            prediction = net.forward(x)
            loss = loss_fn(prediction, y)

            net.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accumulator += loss
            dataset_size += y.size(0)
        accuracy = eval(net, data_loader, device)
        print(f': training loss {loss_accumulator / float(dataset_size):.6f}, accuracy {accuracy}')
        if scheduler is not None:
            scheduler.step()