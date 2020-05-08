import numpy as np
import mrc_functions as mrc
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta as bt
plt.style.use('seaborn-white')

################################
#### We Create the Data set ###

def calculate_example1(sigma=1, n=100):
    beta0=[2,-1,1,0]
    beta0=mrc.normalize(beta0)
    x = np.random.normal(0, 1, [n,4])
    epsilon=sigma*np.random.lognormal(0,1,n)
    Ix=np.dot(x,beta0)
    y=(Ix*epsilon)/(1+np.exp(-Ix+epsilon))

    #### solving the ori
    beta_ini = np.random.uniform(-1, 1, 4)
    beta_ori, tauback, iteration = mrc.ori(y, x, 4, 200, beta_ini)
    Ixback=np.dot(x,beta_ori)


    #solving the glm
    xx = sm.add_constant(x)
    Gaussian_model = sm.GLM(y, xx, family=sm.families.Gaussian())
    Gaussian_results = Gaussian_model.fit()
    beta_glm = mrc.normalize(Gaussian_results.params[1:])

    cosglm=np.abs(np.dot(beta0,beta_glm))
    cosori=np.abs(np.dot(beta0,beta_ori))
    deltacos=cosori-cosglm


    return(cosori, cosglm,deltacos)

trials=1000
n=100
for sigma in [.5]:
    print(sigma)
    dc=np.zeros((trials,3))
    for i in np.arange(trials):
        dc[i,:]=calculate_example1(sigma, n)
        print(sigma, i)

pdc=pd.DataFrame(dc)
pdc.columns=['cos_ori', 'cos_glm', 'delta_cos']
name='data/example1b_'+str(sigma)+'.csv'
pdc.to_csv(name)

pdc=pd.read_csv(name)


plt.hist(pdc['cos_glm'], color='mediumseagreen', label='GLM')
plt.hist(pdc['cos_ori'], color='steelblue', label='ORI')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.savefig('../example_1b_dec2019.png')
plt.show()

