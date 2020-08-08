import numpy as np
import matplotlib.pyplot as plt

np.random.seed(630)
n = 200
noise = 0.5 * np.random.randn(n)
lam = 0.01
x = 3 * (np.random.randn(n, 4) - 0.5)
y = 2 * (2 * x[:, 0] - x[:, 1] + 0.5 + noise > 0) - 1

# steepest gradient descent
num_iter = 1500
w = np.zeros(4)
w_history = []
loss_history = []
lip = max(np.linalg.eig(0.25 * x.T @ x + 2 * lam * np.identity(4))[0])
alpha = 1.0 / lip

for t in range(1, num_iter+1):
    p = 1 / (1 + np.exp(-y * np.dot(x, w)))
    direction = np.sum((1 - p) * y * x.T, axis=1) - 2 * lam * w
    loss = np.sum(np.log(1 + np.exp(-y * np.dot(x, w)))) + lam * np.dot(w, w)
    w_history.append(w)
    loss_history.append(loss)
    w = w + alpha * direction

# newton method
w_n = np.zeros(4)
w_n_history = []
loss_n_history = []

for t in range(num_iter):
    p_n = 1 / (1 + np.exp(-y * np.dot(x, w_n)))
    grad = np.sum((1 - p_n) * (-y) * x.T, axis=1) + 2 * lam * w_n
    hess = (p_n * (1 - p_n) * x.T) @ x + 2 * lam * np.identity(4)
    loss_n = np.sum(np.log(1 + np.exp(-y * np.dot(x, w_n)))) + lam * np.dot(w_n, w_n)
    w_n_history.append(w_n)
    loss_n_history.append(loss_n)
    w_n = w_n - np.dot(np.linalg.inv(hess), grad) * 1.0 / np.sqrt(t+1)

# compare
iter_show = 100

plt.semilogy(loss_history[:iter_show] - loss_n, 'ro-', linewidth=0.5, markersize=1, label='steepest')
plt.semilogy(loss_n_history[:iter_show] - loss_n, 'bo-', linewidth=0.5, markersize=1, label='newton')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('diff of loss')
plt.title('Performance Comparison')

plt.show()