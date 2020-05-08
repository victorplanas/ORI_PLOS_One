import numpy as np
import matplotlib.pyplot as plt
import mrc_functions as mrc
import pandas as pd

beta_ini = mrc.normalize([2, -1, 1, 0])

def data_example_1(n=100, beta0=beta_ini, sigma=.5):
    dim = 4
    if beta0 is None:
        beta0 = [2, -1, 1, 0]
    if sigma<0:
        sigma=0

    beta0 = mrc.normalize(beta0)
    x = np.random.normal(0, 1, [n, dim])
    epsilon = sigma * np.random.lognormal(0, 1, n)
    Ix = np.dot(x, beta0)
    y = (Ix * epsilon) / (1 + np.exp(-Ix + epsilon))
    return x, y, Ix


sigma=10
mean_data=np.zeros((6,2))
sd_data=np.zeros((6,2))
trials=10
dc = np.zeros((trials, 3))

for i in range(trials):
    print(i)
    x, y, Ix = data_example_1(sigma=sigma)
    beta_ori, tau_ori, iter = mrc.ori(y, x, betaini=None, limit=0.00001)
    beta_glm, tau_glm = mrc.glm(y, x)

    Ix_ori = np.dot(x, beta_ori)
    Ix_glm = np.dot(x, beta_glm)

    cor_ori = np.corrcoef(Ix, Ix_ori)[0, 1]
    cor_glm = np.corrcoef(Ix, Ix_glm)[0, 1]
    dc[i, 0] = cor_ori
    dc[i, 1] = cor_glm
    dc[i, 2] = iter

pd_dc=pd.DataFrame(dc)
pd_dc.columns=['cor_ori', 'cor_glm', 'iter_stop']
pd_dc.to_csv('data_april2018/example1.csv', index=False)


