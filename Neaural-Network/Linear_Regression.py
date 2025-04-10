import numpy as np

#Method of Least Squares
def MLS(x: np.ndarray, y:np.ndarray) -> float:
   x_mean = np.mean(x)
   y_mean = np.mean(y)

   return np.sum(x - x_mean) * np.sum(y - y_mean) / np.sum(x - x_mean) ** 2

#Mean Square Error
def MSE(loss: np.ndarray, n: int) -> float:
    return np.sum(loss ** 2) / float(n)

def calcul(a: float, b: float, x: np.ndarray) -> np.ndarray:
    return a * x + b
    
def diff(loss: np.ndarray, n: int, lr: float) -> list:
    d_a = lr * 2 * np.sum(-x * loss) / float(n)
    d_b = lr * 2 * np.sum(-loss) / float(n)

    return d_a, d_b

a = 0
b = 0

lr = 0.03
epoch = 2001

x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

n = np.size(y)

#Gradient Descent
for i in range(epoch):
    y_t = calcul(a, b, x)
    loss = y - y_t
    d = diff(loss, n, lr)

    a -= d[0]
    b -= d[1]

    if i % 100 == 0:
        print("epoch = %.f, a = %.04f, b = %.04f" % (i, a, b))

