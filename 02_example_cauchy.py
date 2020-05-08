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
from scipy.stats import cauchy

n=100
dim=4
sigma=1

def calculate_cauchy(sigma):
    x=np.random.uniform(0,1,(n, dim))
    beta0=[2,1, -1, -1]
    beta0=mrc.normalize(beta0)
    noise= cauchy.rvs(loc=0, scale=.1,size=n)
    I0=np.dot(x,beta0)
    y=I0+sigma*(noise)

    beta_ini = np.random.uniform(-1, -1, dim)
    beta_ori, tauori, iterations = mrc.ori(y, x, 2, 500, beta_ini)
    Ix = np.dot(x, beta_ori)
    beta_ori=mrc.normalize(beta_ori)


    xx = sm.add_constant(x)
    Gaussian_model = sm.GLM(y, xx, family=sm.families.Gaussian())
    Gaussian_results = Gaussian_model.fit()
    #print(Gaussian_results.summary())
    beta_glm = mrc.normalize(Gaussian_results.params[1:])


    cos_glm=np.abs(np.dot(beta0,beta_glm))
    cos_ori=np.abs(np.dot(beta0,beta_ori))
    deltacos=cos_ori-cos_glm
    return(cos_ori, cos_glm,deltacos)


trials=1000
for sigma in [1]:
    print(sigma)
    dc=np.zeros((trials,3))
    for i in np.arange(trials):
        dc[i,:]=calculate_cauchy(sigma)
        print(sigma, i)

    pdc=pd.DataFrame(dc)
    pdc.columns=['cos_ori', 'cos_glm', 'deltacos']
    name='data/new_examplecauchy_'+str(sigma)+'.csv'
    pdc.to_csv(name)


pdc=pd.read_csv('data/new_examplecauchy_1.csv')
plt.figure( dpi=300)
plt.hist(pdc['cos_glm'], color='mediumseagreen', label='GLM')
plt.hist(pdc['cos_ori'], color='steelblue', label='ORI')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.savefig('../figures/exampleCauchy/example_Cauchy.png')
plt.show()

