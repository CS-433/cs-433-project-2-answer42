def eval(net, data_loader, device):
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
    net.to(device)
    for i in range(num_epochs):
        print(f'Epoch {i}', end='')
        net.train()
        loss_accumulator = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            prediction = net.forward(x)
            loss = loss_fn(prediction, y)

            net.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accumulator += loss

        accuracy = eval(net, data_loader, device)
        print(f': training loss {loss_accumulator:.6f}, accuracy {accuracy}')
        if scheduler is not None:
            scheduler.step()