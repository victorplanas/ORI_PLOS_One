
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

sigma=1

xc=np.random.uniform(0,10,(n,4)) #xc = x continuous
beta0=[1,1,1,1]
beta0=mrc.normalize(beta0)
I0=np.dot(xc,beta0)
y=1.5*(I0-8)+np.random.lognormal(0,sigma,n)
y[y < 0] = 0
y[y > 10] = 10

plt.figure(1)
plt.plot(I0, y, 'o', color='steelblue')
plt.xlabel('I(x)')
plt.ylabel('y')
plt.savefig('../compressed_realization_dec2019.png')
plt.show()






