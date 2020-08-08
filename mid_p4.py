import numpy as np
import matplotlib.pyplot as plt
from svm import SVC

np.random.seed(630)
n = 200
lam = 0.01
x = 3 * (np.random.rand(n, 4) - 0.5)
y = (2 * x[:, 0] - 1 * x[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y = 2 * y -1

clf = SVC(kernel="linear", C=1.0)
clf.fit(x, y)

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
    w_history.append(w)
    loss_history.append(hinge_loss)
    grad = np.zeros(4)
    for i in range(len(ywx)):
        if ywx[i] < 1:
            grad += -1
        elif ywx[i] == 1:
            grad += -0.5
    subgrad = np.where(w >= 0, 1, -1)
    w += alpha * (grad + subgrad)

#plt.plot(w_history, hinge_loss, label='hinge loss')
#plt.legend()
#plt.xlim(-2,2)
#plt.ylim(0,2)
#plt.xlabel('w')
#plt.ylabel('loss')
print(hinge_loss)
pred = clf.predict(np.array([-2.76 ,-3.05]).reshape(1, 2))
print(pred)