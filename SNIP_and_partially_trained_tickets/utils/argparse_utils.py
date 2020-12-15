import argparse

def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_normalized(value):
    value = float(value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError("%s is not in range [0, 1]" % value)
    return value