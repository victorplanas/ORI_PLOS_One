
import grad_functions as grd
import mrc_functions as mrc
import matplotlib.pyplot as plt
import pylab
import matplotlib as matplotlib
import random as rd
import numpy as np
from scipy import stats as st
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta as bt
import time

n=100
dim=4

def calculate_examplebehav(sigma=1):
    xc=np.random.uniform(0,10,(n,4)) #xc = x continuous
    beta0=[1,1,1,1]
    beta0=mrc.normalize(beta0)
    I0=np.dot(xc,beta0)
    y=1.5*(I0-8)+np.random.lognormal(0,sigma,n)
    y[y < 0] = 0
    y[y > 10] = 10
    #plt.plot(I0, y, 'o')
    tau_real=st.kendalltau(np.dot(xc,beta0),y)[0]
    x=xc.astype(int)
    #y = y.astype(int)
#   x[abs(x)<2]=0
#   x[abs(x)>5]=5
    tau0=st.kendalltau(np.dot(x,beta0),y)[0]
    I00 = np.dot(x, beta0)
    #plt.plot(I00,y,'o')
    beta_ini = np.random.uniform(-1, 1, dim)
    beta_ori, tauback, iterations = mrc.ori(y, x, 5, 100, beta_ini)
    Ix=np.dot(x,beta_ori)

    xx = sm.add_constant(x)
    Gaussian_model = sm.GLM(y, xx, family=sm.families.Gaussian())
    Gaussian_results = Gaussian_model.fit()
    beta_glm = mrc.normalize(Gaussian_results.params[1:])
    tauglm=st.kendalltau(np.dot(x,beta_glm),y)[0]

    ### STATISTICAL SIGNIFICANCE  PART #####
    # for n=100, d=4, shape1= 8.336884  , shape2= 51.11006  . Think if these estimates apply.
    shape1 = 8.336884
    shape2 = 51.11006
    pval = 1 - bt.cdf(tauback, shape1, shape2, loc=0, scale=1)


    cos_ori=np.dot(beta0,beta_ori)
    cos_glm=np.dot(beta0,beta_glm)
    deltacos=cos_ori-cos_glm
    #plt.plot(Ix,y,'o')
    return(cos_ori, cos_glm, deltacos)

trials=1000

sigma=1

dc = np.zeros((trials, 3))
for i in np.arange(trials):
    print('sigma=', sigma, ' : trial=', i, 'of', trials)
    cos_ori, cos_glm, deltacos=calculate_examplebehav(sigma=.02)
    dc[i,:]=[cos_ori, cos_glm, deltacos]

pdc=pd.DataFrame(dc)
pdc.columns=['cos_ori', 'cos_glm', 'delta_cos']
pdc.to_csv('../example_compressed.csv')

pdc=pd.read_csv('../example_compressed.csv')


plt.hist(pdc['cos_glm'], 30, color='mediumseagreen', label='GLM', alpha=1)
plt.hist(pdc['cos_ori'],30,  color='steelblue', label='ORI', alpha=1)
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.savefig('../example_1b_dec2019.png')
plt.show()
