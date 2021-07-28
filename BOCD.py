#!/usr/bin/env python
# coding: utf-8

# Loading the data from a .mat file into a numpy array

import numpy as np
from scipy.io import loadmat

nmrdata = loadmat('NMRlogWell.mat')
data1 = nmrdata['y'].flatten()
data = np.concatenate((np.zeros(1),data1))

# Visualizing the data

time = np.arange(0, 1101)

import matplotlib.pyplot as plt
plt.plot(time[1:50], data[1:50], marker='.', mfc='red', ms='10')
plt.xlabel('Time')
plt.ylabel('Nuclear Response')
plt.show()


# Defining logarithm of the Posterior Predictive $log(p(x_{t+1}|r_{t}, \textbf{x}_{1:t}))$

from scipy.stats import t as Tdis
from scipy.special import logsumexp

def logtdis(x, mu, k, alpha, beta):
    df = 2*alpha
    nc = mu
    prec = (alpha*k)/(beta*(k+1))
    return Tdis.logpdf(x, df, loc = mu, scale=(prec)**-0.5)


# Initializing the parameters

T = data.size
mu = np.zeros((T+1,T+1))
k = np.zeros((T+1,T+1))
alpha = np.zeros((T+1,T+1))
beta = np.zeros((T+1,T+1))

mu[0,0] = 1.15
k[0,0] = 0.01
alpha[0,0]= 20
beta[0,0] = 2


# Intializing the UPM predictive, Joint Probabilty, Evidence/Normalizing factor and the Run Length Posterior

logUPM = -np.inf*np.ones((T+1,T+1))
logJointP = -np.inf*np.ones((T+1,T+1))
logJointP[0,0] = 0 #setting prior p(r0=0)=1
logZ = np.ones(T)
logRLP = -np.inf*np.ones((T+1,T+1))
RLP = np.zeros((T+1,T+1))


# Defining the Logarithm of the Hazard function

lambdaCP = 250
H = 1/lambdaCP
log_H = np.log(H)
log_1mH = np.log(1-H)


# Bayesian Online Changepoint Detection Algorithm

for t in range(1,T):  
    x = data[t]             #step 2 Observe new datum
    for l in range(t):      #step 3 UPM Predictive Probabilities
        logUPM[t,l] = logtdis(x, mu[t-1,l], k[t-1,l], alpha[t-1,l], beta[t-1,l])
    for l in range(t):      #step 4 Calculating Growth Probabilities
        logJointP[t,l+1] = logJointP[t-1,l] + logUPM[t,l] + log_1mH
    logJointP[t,0] = logsumexp(logJointP[t-1,0:t+1] + logUPM[t,0:t+1] + log_H ) #step 5 Calculating Changepoint Probabilities
    logZ[t] = logsumexp(logJointP, axis=1)[t]       #step 6 Calculating Evidence/Normalizing Factor
    logRLP[t,:] = logJointP[t,:] - logZ[t]          #step 7 Estimating the Run Length Posterior
    mu[t,0] = 1.15          #step 8 Updating Hyperparameters
    k[t,0] = 0.01
    alpha[t,0]= 20
    beta[t,0] = 2
    for l in range(1,t+1):               
        alpha[t,l] = alpha[t-1,l-1] + 0.5
        beta[t,l] = beta[t-1,l-1] + (k[t-1,l-1]*((x-mu[t-1,l-1])**2)/(2*(k[t-1,l-1]+1)))
        k[t,l] = k[t-1,l-1] + 1
        mu[t,l] = ((mu[t-1,l-1]*k[t-1,l-1]) + x)/(k[t-1,l-1] + 1)


# Visualizing the Run Length Posterior distribution

RLP = np.exp(logRLP)

def heatmap2d(arr):
    plt.imshow(arr, cmap='bwr')
    cbar = plt.colorbar()
    cbar.set_label('P(run)', rotation=270)
    plt.ylim(0,1100)
    plt.xlabel('Time')
    plt.ylabel('Run Length')
    plt.show()
    
    
heatmap2d(RLP.transpose())

