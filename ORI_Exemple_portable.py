import numpy as np
import matplotlib.pyplot as plt
import mrc_functions as mrc
import pandas as pd

beta_ini = mrc.normalize([2, -1, 1, 0])



def data_example_1(n=100, beta0=beta_ini, sigma=.5):
    dim = 4
    if beta0 is None:
        beta0 = [2, -1, 1, 0]

    beta0 = mrc.normalize(beta0)
    x = np.random.normal(0, 1, [n, dim])
    epsilon =  np.random.lognormal(0, sigma, n)
    Ix = np.dot(x, beta0)
    y = (Ix * epsilon) / (1 + np.exp(-Ix + epsilon))
    return x, y, Ix

def compute_example(sigma, trials):
    dc = np.zeros((trials, 3))
    for i in range(trials):
        print(sigma, i)
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
    all_ori=dc[:,0]
    all_ori.sort()
    all_glm=dc[:,1]
    all_glm.sort()
    mean_cor_ori=np.mean(dc[:,0])
    mean_cor_glm = np.mean(dc[:, 1])

    return  all_ori, all_glm

sigmas=[.5, 1, 5, 10,20 ,30, 40,  50,60, 70, 80, 90, 100]
d_sigma=len(sigmas)


mean_data=np.zeros((d_sigma,2))
sd_data=np.zeros((d_sigma,2))
top10_data=np.zeros((d_sigma, 2))
worse10_data=np.zeros((d_sigma,2))
trials=500

all_data_ori=np.zeros((trials, d_sigma))
all_data_glm=np.zeros((trials, d_sigma))


for k, sigma in enumerate(sigmas):
    sigma=sigmas[k]
    all_cor_ori, all_cor_glm=compute_example(sigma, trials)
    all_data_ori[:,k]=all_cor_ori
    all_data_glm[:, k] = all_cor_glm
    mean_data[k,:]=[np.mean(all_cor_ori), np.mean(all_cor_glm)]
    worse10_data[k, :] = [np.percentile(all_cor_ori,10), np.percentile(all_cor_glm, 10)]
    top10_data[k, :] = [np.percentile(all_cor_ori,90), np.percentile(all_cor_glm, 90)]

data2save=pd.DataFrame()
data2save['sigma']=sigmas
data2save['ub0_ori']=top10_data[:,0]
data2save['mean_ori']=mean_data[:,0]
data2save['lb0_ori']=worse10_data[:,0]
data2save['ub0_glm']=top10_data[:,1]
data2save['mean_glm']=mean_data[:,1]
data2save['lb0_glm']=worse10_data[:,1]

all_data_ori2save=pd.DataFrame(all_data_ori, columns=sigmas)
all_data_glm2save=pd.DataFrame(all_data_glm, columns=sigmas)


data2save.to_csv('data_april2018/example1_by_sigma.csv', index = False)
all_data_ori2save.to_csv('data_april2018/all_data_ori.csv', index = False)
all_data_glm2save.to_csv('data_april2018/all_data_glm.csv', index = False)

data2save=pd.read_csv('data_april2018/example1_by_sigma.csv')

mean_ori = data2save['mean_ori']
ub0_ori = data2save['ub0_ori']
lb0_ori = data2save['lb0_ori']
mean_glm = data2save['mean_glm']
ub0_glm = data2save['ub0_glm']
lb0_glm = data2save['lb0_glm']

fig, ax = plt.subplots()
len_data = len(mean_ori)
plt.fill_between(sigmas, ub0_ori, lb0_ori, color='orange', alpha=.2)
plt.plot(sigmas, mean_ori, color='orange', label='ORI')

plt.fill_between(sigmas, ub0_glm, lb0_glm, color='blue', alpha=.2)
plt.plot(sigmas,mean_glm, color='blue', label='GLM')

plt.legend(loc='lower left')
ax.set_xlabel('Sigma')
ax.set_ylabel('correlation')
plt.title('Example 1')
plt.savefig('data_april2018/example1.png')
plt.show()



