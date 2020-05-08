import numpy as np
import mrc_functions as mrc
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta as bt
plt.style.use('seaborn-white')

################################
#### We Create the Data set ###

n=100
sigma=.5
beta0=[2,-1,1,0]

beta0=mrc.normalize(beta0)
x = np.random.normal(0, 1, [n,4])
epsilon=sigma*np.random.lognormal(0,1,n)
Ix=np.dot(x,beta0)
y=(Ix*epsilon)/(1+np.exp(-Ix+epsilon))

#### solving the ori
beta_ini = np.random.uniform(-1, 1, 4)
beta_ori, tauback, iteration = mrc.ori(y, x, 4, 200, beta_ini)
Ixori=np.dot(x,beta_ori)


#solving the glm
xx = sm.add_constant(x)
Gaussian_model = sm.GLM(y, xx, family=sm.families.Gaussian())
Gaussian_results = Gaussian_model.fit()
beta_glm = mrc.normalize(Gaussian_results.params[1:])
Ixglm=np.dot(x,beta_glm)

plt.figure(1)
plt.plot(Ixori,y, 'o',color='steelblue')
plt.xlabel('I(x)')
plt.ylabel('y')
plt.savefig('../example_1a_dec2019.png')
plt.show()

