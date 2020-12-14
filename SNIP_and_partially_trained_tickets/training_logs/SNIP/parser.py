import re
import os
import functools
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

dirs = [os.path.join(dir_path, folder) for folder in ['cifar10', 'cifar100']]
files = []
for dir_ in dirs:
    for file_ in os.listdir(dir_):
        files.append((file_, os.path.join(dir_, file_)))

file_name_regex = re.compile(
    r'pruning_(?P<architecture>vgg19|resnet32)_(?P<ratio>\d(\.\d+)?)ratio_(?P<epochs>\d+)epochs_(?P<dataset>cifar10|cifar100)(_(?P<sanitychecks>.+))?\.txt'
)
file_line_regex = re.compile(
    r'Epoch (?P<epoch>\d+): loss (?P<loss>\d(\.\d+)?), train_acc (?P<train_acc>\d+(\.\d+)?)%, test_acc (?P<test_acc>\d+(\.\d+)?)%'
)

df = pd.DataFrame(columns=[
    'architecture', 'sparsity_ratio', 'max_epochs', 'dataset', 'sanity_check', 
    'epoch', 'loss', 'train_acc', 'test_acc'])
for log_file, log_file_path in files:
    m = file_name_regex.match(log_file)
    if m:
        architecture = m.group('architecture')
        sparsity_ratio = m.group('ratio')
        max_epochs = m.group('epochs')
        dataset = m.group('dataset')
        sanitychecks = m.group('sanitychecks')
        if sanitychecks is None:
            sanitychecks = 'original'

        with open(log_file_path, 'r') as f:
            for line in f:
                m = file_line_regex.match(line)
                if m:
                    current_epoch = m.group('epoch')
                    loss = m.group('loss')
                    train_acc = m.group('train_acc')
                    test_acc = m.group('test_acc')

                    df = df.append({
                        'architecture': architecture,
                        'sparsity_ratio': sparsity_ratio,
                        'max_epochs': max_epochs,
                        'dataset': dataset,
                        'sanity_check': sanitychecks,
                        'epoch': current_epoch,
                        'loss': loss,
                        'train_acc': train_acc,
                        'test_acc': test_acc
                    }, ignore_index=True)

df.to_pickle('SNIP_train_logs.pkl')