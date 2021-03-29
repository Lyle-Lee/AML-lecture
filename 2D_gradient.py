import numpy as np
import matplotlib.pyplot as plt

np.random.seed(520)
n = 40
omega = np.random.randn()
noise = 0.8 * np.random.randn(n)
lam = 0.01
x = np.random.randn(n, 2)
y = 2 * (omega * x[:,0] + x[:,1] + noise > 0) - 1

# gradient descent
num_iter = 300
ww = 3
alpha_base = 1
ll_history = []
ww_history = []
lip = 0.25 * max(x[:,0]**2);

for t in range(1,num_iter+1):
    posterior = 1 / (1 + np.exp(-y * (ww * x[:,0] + x[:, 1])));
    direction =  1 / n * np.sum((1 - posterior) * y * x[:,0]) + 2 * lam * ww
    ll = 1/n  * np.sum(np.log(1 + np.exp(-y * (ww * x[:,0] + x[:,1])))) + lam * (ww**2 + 1)
    ww_history.append(ww);
    ll_history.append(ll);
    ww = ww + alpha_base * 1.0 / np.sqrt(t) / lip * direction;

#plt.plot(np.extract(y>0,x[:,0]),np.extract(y>0,x[:,1]), 'x')
#plt.plot(np.extract(y<0,x[:,0]),np.extract(y<0,x[:,1]), 'o')
#plt.plot([-2, -1, 0, 1, 2], [2*ww, ww, 0, -ww, -2*ww], 'b')
#plt.xlabel('x1')
#plt.ylabel('x2')
#plt.title('Data set')

plt.plot(ww_history, ll_history, 'bo-', linewidth=0.5, markersize=0.5, label='steepest')
plt.legend()
plt.xlabel('wight')
plt.ylabel('loss')
plt.title('Gradient Descent')

plt.show()
