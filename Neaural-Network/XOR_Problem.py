import numpy as np

w1 = np.array([[ -2, -2 ], [ 2, 2 ]])
w2 = np.array([ 1, 1 ])
b1 = np.array([ 3, -1 ])
b2 = -1

def Perceptron(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    return Perceptron(np.array([ x1, x2 ]), w1[0], b1[0])

def OR(x1, x2):
    return Perceptron(np.array([ x1, x2 ]), w1[1], b1[1])

def AND(x1, x2):
    return Perceptron(np.array([ x1, x2 ]), w2, b2)

def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

for x in [[ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ]]:
    y = XOR(x[0], x[1])
    print("input : ", x, "output : ", y)