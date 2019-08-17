# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:29:26 2019

@author: Thunderson-RJ
"""

import numpy as np
import matplotlib.pyplot as plt
from classLMS import LMS

#   Definitions:
ensemble    = 100                          # number of realizations within the ensemble
K           = 500;                         # number of iterations
H           = np.array([0.32+0.21j, -0.3+0.7j, 0.5-0.8j, 0.2+0.5j])
Wo          = H                            # unknown system
sigma_n2    = 0.04                         # noise power
N           = 4                            # number of coefficients of the adaptive filter
mu          = .1                           # convergence factor (step)  (0 < mu < 1)

#   Initializing & Allocating memory:
AdaptFilter  = np.zeros([ensemble,K+1,N], dtype = complex)      # coefficient vector for each iteration and realization; w(0) = [0 0 0 0].
MSE          = np.zeros([ensemble,K])                           # MSE for each realization
MSEmin       = np.zeros([ensemble,K])                           # MSE_min for each realization
OutputSignal = np.zeros([ensemble,K], dtype = complex) 
Error        = np.zeros([ensemble,K], dtype = complex)
#   Computing:
for realization in range(ensemble):
    X = np.zeros(N, dtype = complex)
    
    x = (np.sign(np.random.normal(0,1,K)) + 1j*np.sign(np.random.normal(0,1,K)))/np.sqrt(2)
    n = np.random.normal(0,np.sqrt(sigma_n2/2),K)+1j*np.random.normal(0,np.sqrt(sigma_n2/2),K)
    d = np.zeros(K, dtype = complex)
    
    for it in range(K):
        X     = np.roll(X,1)
        X[0]  = x[it]
        d[it] = np.vdot(Wo,X) + n[it]
    
    W = LMS(N-1, [1, 1, 1, 1]) 
    OutputSignal[realization], Error[realization], AdaptFilter[realization] = W.fit(d,x,mu)
    
    MSE[realization]    = np.abs(Error[realization])**2
    MSEmin[realization] = np.abs(n)**2
    
#Averaging
W_av      = np.sum(AdaptFilter,0)/ensemble;
MSE_av    = np.sum(MSE,0)/ensemble;
MSEmin_av = np.sum(MSEmin,0)/ensemble;

#Plotting Learning Curve
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12,6), sharex=True)
ax[0].plot(np.arange(K), 10*np.log10(MSE_av))
ax[0].set_title('Learning Curve for MSE')
ax[0].set_ylabel('MSE [dB]')
ax[0].grid(True)

ax[1].plot(np.arange(K), 10*np.log10(MSEmin_av))
ax[1].set_title('Learning Curve for MSEmin')
ax[1].set_ylabel('MSEmin [dB]')
ax[1].set_xlabel('Number of iterations, k') 
ax[1].grid(True)

#Plotting Coefficients
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12,6), sharex=True)
ax[0].plot(np.arange(K+1), np.real(W_av[:,:]))
ax[0].set_title('Evolution of $W(k)$ (real part)')
ax[0].set_ylabel('Coefficient')
ax[0].grid(True)
ax[0].legend(['Re{$W_0(k)$}', 'Re{$W_1(k)$}', 'Re{$W_2(k)$}', 'Re{$W_3(k)$}'])

ax[1].plot(np.arange(K+1), np.imag(W_av[:,:]))
ax[1].set_title('Evolution of $W(k)$ (imaginary part)')
ax[1].set_ylabel('Coefficient')
ax[1].set_xlabel('Number of iterations, k') 
ax[1].grid(True)
ax[1].legend(['Im{$W_0(k)$}', 'Im{$W_1(k)$}', 'Im{$W_2(k)$}', 'Im{$W_3(k)$}'])

#Plot real part of output signal and input signal of last ensemble
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,6), sharex=True)
ax.plot(np.arange(50), np.real(d)[0:50], label='$d(k)$')
ax.set_title('Desired Signal $d(k)$ vs. Output Signal $Y(k)$')
ax.set_ylabel('Amplitude')
ax.grid(True)

ax.plot(np.arange(50), np.real(OutputSignal[-1, 0:50]), label='$y(k)$')
ax.set_ylabel('Amplitude')
ax.set_xlabel('Number of iterations, k') 
ax.grid(True)
ax.legend(['Desired Sginal Re{d(k)}','Output Signal Re{y(k)}'])