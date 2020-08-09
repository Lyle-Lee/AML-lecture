import numpy as np
import matplotlib.pyplot as plt
import time
from adaboost import AdaBoostClassifier, DecisionStumpClassifier

np.random.seed(808)
n = 40
omega = np.random.randn()
noise = 0.8 * np.random.randn(n)

X = np.random.randn(n, 2)
y = 2 * (omega * X[:, 0] + X[:, 1] + noise > 0) - 1

#plt.plot(np.extract(y>0,X[:,0]),np.extract(y>0,X[:,1]), 'x')
#plt.plot(np.extract(y<0,X[:,0]),np.extract(y<0,X[:,1]), 'o')

Estimators = 60
Iteration_Steps = 60
trainning_time = 0
boost = AdaBoostClassifier(Estimators, DecisionStumpClassifier(Iteration_Steps))
tbegin = time.time()
boost.train(X, y)
tend = time.time()
trainning_time = tend - tbegin

plot_step = 0.02
fig = plt.figure(figsize=(5, 5))

# Plot the decision boundaries
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, plot_step),
                     np.arange(x2_min, x2_max, plot_step))

Z, _ = boost.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the trainning points
idx = np.where(y == 1)
plt.scatter(X[idx, 0], X[idx, 1],
                color='b', cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class +1")
ido = np.where(y == -1)
plt.scatter(X[ido, 0], X[ido, 1],
                color='r', cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class -1")
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.legend(loc='upper right')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary')

yPred, CI = boost.predict(X)
accuracy = np.mean(yPred == y)

fig.canvas.set_window_title('AdaBoost Test' + ' - '
    + 'estimators: %d, iteration steps: %d, accuracy: %0.3f, trainning time: %0.4f' %
    (Estimators, Iteration_Steps, accuracy, trainning_time))

plt.show()