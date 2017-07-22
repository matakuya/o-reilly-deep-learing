import numpy as np

def softmax(a):
    c = np.max(a)
    # To take measures against an overflow.
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
