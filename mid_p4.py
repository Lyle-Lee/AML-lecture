import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
#from svm import SVC

np.random.seed(630)
n = 200
features = 4
noise = 0.5 * np.random.randn(n)
lam = 0.01
x = 3 * (np.random.randn(n, 4) - 0.5)
y = 2 * (2 * x[:, 0] - x[:, 1] + 0.5 + noise > 0) - 1
#x = np.matrix(np.random.normal(size=n * features).reshape(n, features))  # gausian distributed
#y = 2 * (x.sum(axis=1) > 0) - 1.0

# experiment with svm API
#clf = SVC(kernel="linear", C=1.0)
#clf.fit(x, y)
#pred = clf.predict(np.array([-2.76 ,-3.05]).reshape(1, 2))
#print(pred)

# cvx
w_lasso = cv.Variable((4,1))
obj_fn = cv.sum(cv.pos(1 - cv.multiply(y.reshape(200, 1), x @ w_lasso))) +  lam * cv.norm(w_lasso, 1)
objective = cv.Minimize(obj_fn)
#constraints = []
prob = cv.Problem(objective)
result = prob.solve(solver=cv.CVXOPT) 
w_lasso = w_lasso.value

# batch proximal subgradient
num_iter = 1500
w = np.zeros(4)
w_history = []
loss_history = []
lip = max(np.linalg.eig(0.25 * x.T @ x + 2 * lam * np.identity(4))[0])
alpha = 1.0 / lip

for t in range(num_iter):
    reg = lam * np.sum(np.abs(w))
    ywx = y * np.dot(x, w)
    hinge_loss = np.sum(np.maximum(np.zeros(n), 1 - ywx)) + reg
    grad = np.zeros(4)
    for i in range(n):
        if ywx[i] < 1:
            grad += -(y * x.T).T[i]
        elif ywx[i] == 1:
            grad += -0.5 * (y * x.T).T[i]
    g = np.where(w > 0, 1, w)
    g = np.where(g < 0, -1, g)
    subgrad = grad + np.where(g == 0, 0, g)
    w_history.append(w)
    loss_history.append(hinge_loss)
    w = w + alpha * (-subgrad)

plt.plot(np.abs(w_history - w_lasso.reshape(-1)), label='hinge loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('|w_prox - w_cvx|')
plt.show()