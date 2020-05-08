import numpy as np
import grad_functions as grd
import mrc_functions as mrc
import matplotlib.pyplot as plt
import pylab
import matplotlib as matplotlib
import random as rd
from scipy import stats as st
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta as bt
import seaborn as sns

################################
#### We Create the Data set ###


def mixturenoise(klimit, mu1, sigma1, mu2, sigma2, n):
    epsilon=np.zeros(n)
    for i in np.arange(n):
        k=np.random.uniform(0,1,1)
        if k<klimit:
             mu, sigma=[mu1, sigma1]
        else:
            mu, sigma = [mu2, sigma2]
        epsilon[i]=np.random.normal(mu, sigma,1)
    return(epsilon)

sigma=1
n=200
dim=4
beta0=[2,-1,1,0]
beta0=mrc.normalize(beta0)
x = np.random.normal(3, 1, [n,4])
epsilon=sigma*mixturenoise(.3, 2, 1, -2, .5, n)
Ix=np.dot(x,beta0)
y=(Ix+epsilon)



beta_ini = np.random.uniform(-1, 1, dim)
beta_ori, tauback, iterations = mrc.ori(y, x, 4, 200, beta_ini)
Ixback=np.dot(x,beta_ori)

plt.figure(dpi=300)
plt.plot(Ixback, y, 'o', color='steelblue')
plt.xlabel('I(x)')
plt.ylabel('y')
plt.savefig('../figures/example_mixture/mixture_realization_dec2019.png')
plt.show()



