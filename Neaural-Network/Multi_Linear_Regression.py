import numpy as np

#Method of Least Squares
def MLS(x: np.ndarray, y:np.ndarray) -> list:
    x_mean = []
   
    for x_i in x:
        x_mean.append(np.mean(x_i))

    y_mean = np.mean(y)
    print(x_mean)

    a_t = []
    for i in range(len(x_mean)):
        a_t.append(np.sum(x[i] - x_mean[i]) * np.sum(y - y_mean) / np.sum(x[i] - x_mean[i]) ** 2)
        print(np.sum(x[i] - x_mean[i]))
        print(np.sum(y - y_mean))
        print(np.sum(x[i] - x_mean[i]) ** 2)

    return a_t

#Mean Square Error
def MSE(loss: np.ndarray, n: int) -> float:
    return np.sum(loss ** 2) / float(n)

def calcul(a: list, b: float, x: np.ndarray) -> np.ndarray:
    x_t = x.copy()
    for i in range(np.ndim(x)):
        x_t[i] *= a[i]

    return np.sum(x_t, axis = 0) + b
    
def diff(loss: np.ndarray, n: int, lr: float) -> list:
    d_a = []
    for i in range(x.ndim):
        d_a.append(lr * 2 * np.sum(-x[i] * loss) / float(n))

    d_b = lr * 2 * np.sum(-loss) / float(n)

    return d_a, d_b

a = [0, 0]
b = 0

lr = 0.01
epoch = 2001

x = np.array([[2.0, 4.0, 6.0, 8.0], [0.0, 4.0, 2.0, 3.0]])
y = np.array([81, 93, 91, 97])

a = MLS(x, y)

print(a)

n = np.size(y)

# y_t = calcul(a, b, x)
# loss = y - y_t
# d = diff(loss, n, lr)
# print(d)

#Gradient Descent
for i in range(epoch):
    y_t = calcul(a, b, x)
    loss = y - y_t
    d = diff(loss, n, lr)

    for j in range(len(a)):
        a[j] -= d[0][j]
    b -= d[1]

    if i % 100 == 0:
        print("epoch = %.f, a_1 = %.04f, a_2 = %.04f, b = %.04f" % (i, a[0], b))

