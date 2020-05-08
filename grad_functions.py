import numpy as np
import mrc_functions as mrc
import matplotlib.pyplot as plt
from scipy import stats as st
import pandas as pd


def createdata(n, dim, beta, sd):
    beta=beta/np.sqrt(np.dot(beta, beta))
    x=np.random.uniform(-1,0,(n,dim))
  #  x=np.random.normal(0,1,(n, dim))
    Ix=np.dot(x,beta)
    noise=sd*(np.random.normal(0,1,n))
#    y=2*Ix+noise+1
#    y=np.exp(Ix+noise)*(Ix**2-1)
    y=Ix + np.abs(np.sin(2 * Ix+noise))
    plt.plot(Ix, y, 'o')
    return(x,y)

def concordance_matrix(x,y,beta=1,theta=None, clipping=30):
    n=len(y)
    Ix=np.dot(x,beta)
    Ixt=Ix.transpose()
    y=np.array(y)
    yt=y.transpose()
    Ix.shape=(n,1)
    Ixt.shape=(1,n)
    y.shape=(n,1)
    yt.shape=(1,n)
    eta = (Ix - Ixt) * (y - yt)
    eta_logical = 1 * (eta > 0)
    if theta==None:
        fij=eta_logical
    else:
        eta_clip=np.clip(theta*eta,-clipping,clipping)
        fij=np.exp(eta_clip)/(1+np.exp(eta_clip))
    return(fij)

def concordance_point(x,y,i, beta=1,theta=None, clipping=30):
    n=len(y)
    fij=concordance_matrix(x,y,beta, theta, clipping)
    loc_concord=sum(fij[i,:])/n
    return(loc_concord)

def tau_point(x,y,i, beta=1,theta=None, clipping=30 ):
    cp=concordance_point(x,y,i, beta,theta,clipping )
    tau_point=2*cp-1
    return(tau_point)

def concordance_global(x,y, beta=1,theta=None, clipping=30 ):
    n=len(y)
    fij=concordance_matrix(x,y,beta, theta, clipping)
    glob_concord=np.sum(fij)/(n*(n-1))
    return(glob_concord)

def kendalltau(x,y, beta=1,theta=None, clipping=30 ):
    glob_concord=concordance_global(x,y, beta,theta, clipping )
    kendalltau=2*glob_concord-1
    return(kendalltau)

def orthonormal(input_v,base_v):
    v_orthogonal=input_v-np.dot(input_v, base_v)*base_v
    v_orthonormal=mrc.normalize(v_orthogonal)
    return(v_orthonormal)


def gradient_i(x,y,i,beta,theta,clipping=30 ):
    n=len(y)
    yt=y
    yt.shape=(n,1)
    Fi_t=concordance_matrix(x,y,beta,theta, clipping)[i]
    Fi_t.shape=(n,1)
    eta_matrix=-theta*(x[i,:]-x)*(yt[i]-y)*(Fi_t)*(1-Fi_t)
    eta=np.sum(eta_matrix, axis=0)/(n-1)
    eta_orth=orthonormal(eta, beta)
    return(eta_orth)

def step_gradient(x,y,i,beta,theta, learning_rate=1):
    beta_next=beta-learning_rate* gradient_i(x,y,i,beta,theta)
    beta_next=mrc.normalize(beta_next)
    return(beta_next)

def Openball(x,y,x0, alpha=.8):
    if alpha==1:
        xselect=x
        yselect=y
        return (xselect, yselect)
    else:
        n=len(x[:,0])
        dist = np.sum((x - x0) ** 2, axis=1)
        distsort = np.array(dist.copy())
        distsort.sort()
        lim_dist = distsort[int(n * alpha)]
        xselect = x[dist < lim_dist,]
        yselect=y[dist<lim_dist]
        return(xselect, yselect)


def gradient_descent(x,y,i, beta=None, theta=None,  learning_rate=2, steps_max=1000, threshold=.0001, alpha=None):
    dim=len(x[0,])
    if beta==None:
        beta=np.random.uniform(-1,1,dim)
    if alpha==None:
        ib=i
    else:
        x0=x[i,]
        xb,yb= Openball(x,y,x0,alpha)
        ib=np.where(xb==x[i,])[0][0]
    if theta==None:
        sdmax=max(np.std(x, axis=0))
        theta=4*sdmax
    trials = steps_max
    betadisc = np.zeros((trials, dim))
    taudisc = np.zeros(trials) - 1
    beta_t = mrc.normalize(np.random.uniform(-1, 1, dim))
    for k in np.arange(1, trials):
        beta_t = step_gradient(x, y, ib, beta_t, theta, learning_rate)
        betadisc[k] = beta_t
        taudisc[k] = st.kendalltau(np.dot(x, betadisc[k, :]), y)[0]
        if taudisc[k] < taudisc[k - 1]:
            learning_rate = learning_rate/2
            theta=theta+.1
            #print('new lr at k=', k, " lr=", learning_rate)
            beta_t = betadisc[k - 1]
            betadisc[k] = betadisc[k - 1]
            taudisc[k] = taudisc[k - 1]
            continue
        if k > 10:
            if abs(taudisc[k] - taudisc[k - 3]) < threshold:
                k_final = k
                #print('convergence at k=', k_final)
                break

    return(betadisc[k], taudisc[k], k)