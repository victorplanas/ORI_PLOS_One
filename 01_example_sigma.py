import numpy as np
import matplotlib.pyplot as plt
import mrc_functions as mrc
import pandas as pd
import seaborn as sns


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

def graph_example1():
    x,y,Ix=data_example_1(100,beta_ini, 1)
    f=plt.figure()
    plt.plot(Ix,y,'o')
    f.show()
    f.savefig('foo.pdf')


def compute_example(sigma, trials):
    dcorr = np.zeros((trials, 3))
    dcos=np.zeros((trials, 3))
    for i in range(trials):
        print(sigma, i)
        x, y, Ix = data_example_1(sigma=sigma)
        d = x.shape[1]
        betaini = mrc.normalize(np.random.uniform(-1, 1, d))
        beta_ori, tau_ori, iter = mrc.ori(y, x, betaini=betaini, limit=0.0001)
        beta_glm, tau_glm = mrc.glm(y, x)

        Ix_ori = np.dot(x, beta_ori)
        Ix_glm = np.dot(x, beta_glm)

        cor_ori = np.corrcoef(Ix, Ix_ori)[0, 1]
        cor_glm = np.corrcoef(Ix, Ix_glm)[0, 1]

        cos_ori=np.abs(np.dot(beta_ori, beta_ini))
        cos_glm=np.abs(np.dot(beta_glm, beta_ini))

        dcorr[i, 0] = cor_ori
        dcorr[i, 1] = cor_glm
        dcorr[i, 2] = iter

        dcos[i, 0] = cos_ori
        dcos[i, 1] = cos_glm
        dcos[i, 2] = iter




    all_ori=dcorr[:,0]
    all_glm=dcorr[:,1]
    all_iter=dcorr[:,2]

    return  dcorr, dcos



def plot_ci(data2save, lab='Cosine Similarity'):
    mean_ori = data2save['mean_ori']
    ub0_ori = data2save['ub0_ori']
    lb0_ori = data2save['lb0_ori']
    mean_glm = data2save['mean_glm']
    ub0_glm = data2save['ub0_glm']
    lb0_glm = data2save['lb0_glm']
    fig, ax = plt.subplots( dpi=300)
    len_data = len(mean_ori)
    plt.fill_between(sigmas, ub0_ori, lb0_ori, color='steelblue', alpha=.2)
    plt.plot(sigmas, mean_ori, color='steelblue', label='ORI')

    plt.fill_between(sigmas, ub0_glm, lb0_glm, color='mediumseagreen', alpha=.2)
    plt.plot(sigmas,mean_glm, color='mediumseagreen', label='GLM')

    plt.legend(loc='lower left')
    ax.set_xlabel('Sigma')
    ax.set_ylabel(lab)
    plt.savefig('../example1_sigma.png')
    plt.show()
    return fig

def dataCI(mean_data, worse10_data, top10_data):
    data2save=pd.DataFrame()
    data2save['sigma']=sigmas
    data2save['ub0_ori']=top10_data[:,0]
    data2save['mean_ori']=mean_data[:,0]
    data2save['lb0_ori']=worse10_data[:,0]
    data2save['ub0_glm']=top10_data[:,1]
    data2save['mean_glm']=mean_data[:,1]
    data2save['lb0_glm']=worse10_data[:,1]

    return data2save

#sigmas=[1,2, 5, 10, 20,30,  40,50,  60,70,  80, 90, 100]
sigmas=[1,2, 5, 10, 20, 40]
d_sigma=len(sigmas)


mean_data_corr=np.zeros((d_sigma,2))
top10_data_corr=np.zeros((d_sigma, 2))
worse10_data_corr=np.zeros((d_sigma,2))

mean_data_cos=np.zeros((d_sigma,2))
top10_data_cos=np.zeros((d_sigma, 2))
worse10_data_cos=np.zeros((d_sigma,2))


trials=100
#sd_data=np.zeros((d_sigma,2))


all_data_ori_corr=np.zeros((trials, d_sigma))
all_data_glm_corr=np.zeros((trials, d_sigma))
all_data_ori_cos=np.zeros((trials, d_sigma))
all_data_glm_cos=np.zeros((trials, d_sigma))


for k, sigma in enumerate(sigmas):
    dcorr, dcos=compute_example(sigma, trials)
    #all_cor_ori, all_cor_glm, all_iter=compute_example(sigma, trials)

    all_data_ori_corr[:,k]=dcorr[:,0]
    all_data_glm_corr[:, k] = dcorr[:,1]
    all_data_ori_cos[:,k]=dcos[:,0]
    all_data_glm_cos[:, k] = dcos[:,1]

    mean_data_corr[k,:]=[np.mean( all_data_ori_corr[:,k]), np.mean( all_data_glm_corr[:,k])]
    worse10_data_corr[k, :] = [np.percentile(all_data_ori_corr[:,k],10), np.percentile(all_data_glm_corr[:,k], 10)]
    top10_data_corr[k, :] = [np.percentile(all_data_ori_corr[:,k],90), np.percentile(all_data_glm_corr[:,k], 90)]

    mean_data_cos[k,:]=[np.mean( all_data_ori_cos[:,k]), np.mean( all_data_glm_cos[:,k])]
    worse10_data_cos[k, :] = [np.percentile(all_data_ori_cos[:,k],10), np.percentile(all_data_glm_cos[:,k], 10)]
    top10_data_cos[k, :] = [np.percentile(all_data_ori_cos[:,k],90), np.percentile(all_data_glm_cos[:,k], 90)]

    all_data_ori_corr2save=pd.DataFrame(all_data_ori_corr, columns=sigmas)
    all_data_glm_corr2save=pd.DataFrame(all_data_glm_corr, columns=sigmas)
    all_data_ori_cos2save=pd.DataFrame(all_data_ori_cos, columns=sigmas)
    all_data_glm_cos2save=pd.DataFrame(all_data_glm_cos, columns=sigmas)




data2save_corr=dataCI(mean_data_corr, worse10_data_corr, top10_data_corr)
data2save_corr.to_csv('../example1_data2save_corr.csv', index = False)

data2save_cos=dataCI(mean_data_cos, worse10_data_cos, top10_data_cos)
data2save_cos.to_csv('../example1_data2save_cos.csv', index = False)



data2save_cos=pd.read_csv('../example1_data2save_cos.csv')

#plot_ci(data2save_corr, lab='Correlation')
plot_ci(data2save_cos, lab='Cosine Distance')
