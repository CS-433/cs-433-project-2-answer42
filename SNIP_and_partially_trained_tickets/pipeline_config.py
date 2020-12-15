from models import resnet32, vgg19
from utils.dataloading_utils import load_cifar10, load_cifar100, create_loader, split_features_and_labels

supported_architectures = {
    'resnet32': resnet32,
    'vgg19': vgg19
}
TEST_BATCH_SIZE = 1024
TRAINING_BATCH_SIZE = 64
NUM_WORKERS = 4
datasets = {
    'cifar10': {
        'data': load_cifar10,
        'num_classes': 10
    },
    'cifar100': {
        'data': load_cifar100,
        'num_classes': 100
    }
}
SMART_RATIO_CONFIG = {
    'resnet32': {
        'L': 33,
        'vgg_scaling': False
    },
    'vgg19': {
        'L': 19,
        'vgg_scaling': True
    }
}