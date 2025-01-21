import numpy as np
import matplolib.pyplot as plt

with open("../tmp.txt", 'r') as f:
    data = list(map(dict, f.readlines()))
