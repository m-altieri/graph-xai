import numpy as np


def obs2seqs(data, h, p, stride):
    obs, nodes, features = data.shape
    assert obs % stride == 0  # Check there are no partial sequences

    seqs = (obs - h - p) // stride + 1
    x = np.zeros((seqs, h, nodes, features))
    y = np.zeros((seqs, p, nodes, features))
    for i in range(seqs):
        x[i] = data[i * stride: i * stride + h]
        y[i] = data[i * stride + h: i * stride + h + p]  # y is the next P steps after x

    return x, y
