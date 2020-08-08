import numpy as np
import matplotlib.pyplot as plt

np.random.seed(630)
C = 3
n = 200
x = 3 * (np.random.rand(n, 4) - 0.5)
W = np.array([[ 2,  -1, 0.5],
              [-3,   2,   1],
              [ 1,   2,   3]])
y = np.argmax(np.dot(np.hstack([x[:,:2], np.ones((n, 1))]), W.T)
                        + 0.5 * np.random.randn(n, 3), axis=1)
lam = 0.01

# steepest gradient descent
num_iter = 600
w = np.zeros((C, 4))
direction = []
w_history = []
loss_history = []
lip = max(np.linalg.eig(0.25 * x.T @ x + 2 * lam * np.identity(4))[0])
alpha = 1.0 / lip

for t in range(1, num_iter+1):
    p = np.exp(w @ x.T) / np.sum(np.exp(x @ w.T), axis=1)
    loss = 0
    for i in range(C):
        direction.append(np.sum(((y == i) - p[i]) * x.T, axis=1) - 2 * lam * w[i])
        loss += np.sum((y == i) * (np.log(np.sum(np.exp(x @ w.T), axis=1)) - np.dot(x, w[i]))) + lam * np.dot(w[i], w[i])
    w_history.append(w)
    loss_history.append(loss)
    w = w + 1.0 / np.sqrt(t) * alpha * np.array(direction)
    direction.clear()

# newton method
w_n = np.zeros((C, 4))
grad = []
w_n_history = []
loss_n_history = []

for t in range(num_iter):
    p_n = np.exp(w_n @ x.T) / np.sum(np.exp(x @ w_n.T), axis=1)
    loss_n = 0
    hess = np.empty((0, C*4))
    for c in range(C):
        grad.append(np.sum((p_n[c] - (y == c)) * x.T, axis=1) + 2 * lam * w_n[c])
        loss_n += np.sum((y == c) * (np.log(np.sum(np.exp(x @ w_n.T), axis=1)) - np.dot(x, w_n[c]))) + lam * np.dot(w[c], w[c])
        h = np.empty((4, 0))
        for k in range(C):
            h = np.hstack([h, (p_n[c] * ((k == c) - p_n[k]) * x.T) @ x + (k == c) * 2 * lam * np.identity(4)])
        hess = np.vstack([hess, h])
    w_n_history.append(w_n)
    loss_n_history.append(loss_n)
    w_n = (w_n.reshape(-1) - np.dot(np.linalg.inv(hess), np.array(grad).reshape(-1)) * 1.0 / np.sqrt(t+1)).reshape((C, 4))
    grad.clear()

# compare
iter_show = 200

plt.semilogy(np.abs(loss_history[:iter_show] - loss_n), 'ro-', linewidth=0.5, markersize=1, label='steepest')
plt.semilogy(np.abs(loss_n_history[:iter_show] - loss_n), 'bo-', linewidth=0.5, markersize=1, label='newton')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('diff of loss')
plt.title('Performance Comparison (multiclass)')

plt.show()