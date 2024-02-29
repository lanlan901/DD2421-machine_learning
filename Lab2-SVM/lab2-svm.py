import math
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import minimize

# 设置随机种子以确保结果可重复
np.random.seed(100)


# TODO 01: define a kernel function

# 线性核函数
def linear_kernel(x, y):
    return np.dot(x, y)


# 多项式核函数
def polynomial_kernel(x, y, p=2):
    return (np.dot(x, y) + 1) ** p


# RBF核函数
def rbf_kernel(x, y, sigma=1.5):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


# TODO 02: Implement the function objective based on equation (4)

# 最小化关于拉格朗日乘子αi的函数：
# 所有样本对（i,j）的拉格朗日乘子、标签和核函数值的乘积之和的1/2 - 所有拉格朗日乘子的和

def objective(alpha, P):
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)


# TODO 03:  Implement the function zerofun based on equation (10)
def zerofun(alpha, target):
    return np.dot(alpha, target)


def train_svm(X, target, kernel):
    N = len(X)
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = target[i] * target[j] * kernel(X[i], X[j])

    start = np.zeros(N)
    C = 40
    B = [(0, C) for _ in range(N)]
    XC = {'type': 'eq', 'fun': lambda a: zerofun(a, target)}

    ret = minimize(fun=lambda a: objective(a, P), x0=start, bounds=B, constraints=XC)
    alpha = ret['x']

    # TODO 04: Extract the non-zero α values
    # 提取非零α值
    threshold = 1e-5
    non_zero_alphas = [(a, X[i], target[i]) for i, a in enumerate(alpha) if threshold < a < C]
    b = compute_b(non_zero_alphas, kernel)

    return non_zero_alphas, b


# TODO 05: Calculate the b value using equation (7)
def compute_b(non_zero_alphas, kernel):
    s = non_zero_alphas[0][1]
    t_s = non_zero_alphas[0][2]
    b_sum = sum(alpha * t * kernel(x, s) for alpha, x, t in non_zero_alphas)
    b = b_sum / len(non_zero_alphas) - t_s
    return b



# TODO 06: indicator function based on equation(6)
def indicator_function(x, non_zero_alphas, b, kernel):
    indicator_result = sum(alpha * t * kernel(x_i, x) for alpha, x_i, t in non_zero_alphas) - b
    return indicator_result


# 数据
classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [- 1.5, 0.5],
                         np.random.randn(10, 2) * 0.2 + [1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
inputs = np.concatenate((classA, classB))
# 标签
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

# 获取数据点的总数
N = inputs.shape[0]
# 随机打乱数据点
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# plot
plt.figure(figsize=(8, 6))
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label='Class A')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label='Class B')
plt.axis('equal')
plt.title('Training Data with Class A and Class B')
plt.legend()
plt.show()
# train SVM
non_zero_alphas, b = train_svm(inputs, targets, rbf_kernel)
# plot decision boundary
xgrid = np.linspace(-5, 5, 50)
ygrid = np.linspace(-4, 4, 50)
grid = np.array([[indicator_function(np.array([x, y]), non_zero_alphas, b, rbf_kernel)
                  for x in xgrid]
                 for y in ygrid])

plt.figure(figsize=(12, 8))
contour = plt.contour(xgrid, ygrid, grid, levels=(-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
plt.clabel(contour, inline=1, fontsize=10)
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label='Class A')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label='Class B')

for alpha, x, t in non_zero_alphas:
    plt.plot(x[0], x[1], 'yo')

plt.title('SVM Decision Boundary with Support Vectors')
plt.axis('equal')
plt.legend()
plt.savefig('svmcontour.pdf')
plt.show()

