import numpy as np
from scipy import stats as st
import statsmodels.api as sm


def normalize(beta):
    if np.dot(beta, beta) > 0:
        beta_normal = beta / np.sqrt(np.dot(beta, beta))
    else:
        beta_normal = beta
    return beta_normal


'''
unused function
def initialize(beta, d, n):
    # definition of constants of function
    K = 10
    sigma = 1
    # here i create new x each time. to generate Null model histograms
    x = np.random.normal(0, 1, (n, d))
    epsilon = np.random.normal(0, 1, n)
    Ix = np.matmul(x, beta)
    y = (K * (Ix + sigma * epsilon) / (1 + np.exp(Ix + sigma * epsilon)))
    return y, Ix
'''

def newbeta(d):
    betatmp = np.random.uniform(-1, 1, d)
    beta = betatmp / np.sqrt(np.dot(betatmp, betatmp))
    return beta


def trialbeta(x, thisy, d, mc):
    taumax = -1
    betamax = []
    for i in range(0, mc):
        beta = newbeta(d)
        # print(beta)
        Ix = np.matmul(x, beta)
        z = (st.kendalltau(Ix, thisy))[0]
        if z > taumax:
            taumax = z
            betamax = beta
    return betamax


def PartialOptimBeta(x, y, betaini, pos, gridsize):
    betamax = betaini.copy()
    Ix = np.matmul(x, betaini)
    taumax = (st.kendalltau(Ix, y))[0]
    betatmp = betaini.copy()  # empty array to story my winning beta
    iniscreening=0
    endscreening=1
    for i in np.arange(iniscreening, endscreening, 1/gridsize):
        betatmp[pos] = e_logit(i)
        betatmpn = normalize(betatmp)
        if (np.dot(betatmpn, betatmpn) != 0):
            Ixtmp = np.matmul(x, betatmpn)
            z = (st.kendalltau(Ixtmp, y))[0]
            if z > taumax:
                taumax = z
                betamax = betatmpn.copy()
    betamax = normalize(betamax)
    return betamax, taumax


def e_logit(p, epsilon=.0001):
    return (np.log((p + epsilon) / (1 - p + epsilon)))


def ori(y, x, iterations_max=10, gridsize=1000, betaini=None, limit=0.0001):
    tautmp_pre=1
    n = len(y)
    d=x.shape[1]
    if betaini is None:
        betaini=normalize(np.random.uniform(-1,1,d))

    for iteration in range(iterations_max):

        for pos in range(d):

            updateBeta, tautmp = PartialOptimBeta(x, y, betaini, pos, gridsize)
            betaini = updateBeta.copy()

        delta_tau = abs(tautmp-tautmp_pre)/tautmp_pre
        #print(delta_tau )
        if delta_tau<limit:
            break
        tautmp_pre = tautmp


    return updateBeta, tautmp, iteration

def glm(y,x):
    xx = sm.add_constant(x)
    Gaussian_model = sm.GLM(y, xx, family=sm.families.Gaussian())
    Gaussian_results = Gaussian_model.fit()
    beta_glm = normalize(Gaussian_results.params[1:])
    Ix_glm=np.matmul(x,beta_glm)
    tau_glm = (st.kendalltau(Ix_glm, y))[0]

    return beta_glm, tau_glm

def stepwise(y,x,iterations=5, gridsize=100, betaini=None ):
    n=len(y)
    dim=x.shape[1]
    if dim<=2:
        print('dim<=2')
        quit()
    else:
        tau_red = np.zeros(dim)
        beta_red = np.zeros((dim, dim - 1))
        for k in np.arange(dim):
            print(k)
            keep = set(np.arange(dim)) - set([k])
            xk = x[:, list(keep)]
            beta_ini = np.random.uniform(-1, 1, xk.shape[1])
            betaback_k, tauback_k = ori(y, xk, iterations, gridsize, betaini)
            tau_red[k] = tauback_k
            beta_red[k, :] = betaback_k


    return(beta_red, tau_red)

def betaglm(y,x):
    xx = sm.add_constant(x)
    Gaussian_model = sm.GLM(y, xx, family=sm.families.Gaussian())
    Gaussian_results = Gaussian_model.fit()
    beta_glm = normalize(Gaussian_results.params[1:])
    return(beta_glm)

def tauglm(y,x):
    beta_glm = normalize(betaglm(y,x))
    tau_glm=st.kendalltau(np.dot(x,beta_glm),y)[0]
    return(tau_glm)
