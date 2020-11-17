from models import resnet
from models import main
from models import schedulers


REGISTERED_MODELS = {
    'basic-supervised': main.BasicSupervised,
}
