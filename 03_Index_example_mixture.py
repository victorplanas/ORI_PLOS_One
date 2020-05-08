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

def calculate_examplemixtures(sigma=1):
    n=200
    dim=4
    beta0=[2,-1,1,0]
    beta0=mrc.normalize(beta0)
    x = np.random.normal(3, 1, [n,4])
    epsilon=sigma*mixturenoise(.3, 2, 1, -2, .5, n)
    Ix=np.dot(x,beta0)
    y=(Ix+epsilon)
    #‹


    #### solving the ori
    beta_ini = np.random.uniform(-1, 1, dim)
    beta_ori, tauback, iterations = mrc.ori(y, x, 4, 200, beta_ini)
    Ixback=np.dot(x,beta_ori)
    #plt.plot(Ixback, y, 'o')
    #print(betaback)
    #print(tauback)

    #solving the glm
    xx = sm.add_constant(x)
    Gaussian_model = sm.GLM(y, xx, family=sm.families.Gaussian())
    Gaussian_results = Gaussian_model.fit()
    beta_glm = mrc.normalize(Gaussian_results.params[1:])
    #plt.plot(np.dot(x,beta_glm),y,'o')

    cos_glm=np.abs(np.dot(beta0,beta_glm))
    cos_ori=np.abs(np.dot(beta0,beta_ori))
    deltacos=cos_ori-cos_glm
    return(cos_ori, cos_glm, deltacos )


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

trials=1000
sigma=1
dc=np.zeros((trials, 3))
for i in np.arange(trials):
    print(i)
    dc[i,:]=calculate_examplemixtures(sigma)

pdc=pd.DataFrame(dc)
pdc.columns=['cos_ori', 'cos_glm', 'deltacos']
name='../mixture_dec2019.csv'
pdc.to_csv(name)

pdc=pd.read_csv('../mixture_dec2019.csv')

plt.figure()
plt.hist(pdc['cos_glm'], 30,color='mediumseagreen', label='GLM')
plt.hist(pdc['cos_ori'],30, color='steelblue', label='ORI')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.savefig('../mixture_dec2019.png')
plt.show()



