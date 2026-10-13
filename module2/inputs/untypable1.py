import numpy as np


def split_and_count(flags):
    flags.partition(0)
    return flags.bit_length()
