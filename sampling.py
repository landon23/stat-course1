#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:59:35 2018

@author: benlandon
"""

import numpy as np
import time
from scipy.stats import norm


"""
This function uses rejection sampling to draw samples from a posterier distribution.

Takes in a multinomial sample, and a time and max_samples.
"""

def rej_samples(Y,maxsamp=1000000,maxtime=20, verbose = True):
    samples = np.array([])
    timein = time.time()
    timeout = time.time() + maxtime
    logM = calclogM(Y)
    iterations = 0
    while (samples.size < maxsamp) & (time.time() < timeout):
        iterations = iterations + 1
        U = np.log(np.random.uniform())
        X = np.random.uniform()
        if U < logpx(Y,X)-logM:
            samples = np.append(samples,X)
    if (time.time() >= timeout) & verbose:
        print('Max time reached.', samples.size, 'samples were generated.')
    if (samples.size >= maxsamp) & verbose:
        print('Max samples reached.  This took', "{0:.2f}".format(time.time()-timein), 'seconds.')
    if verbose:
        print('Inner while loop iterations:', iterations)
    return samples

        
    
def logpx(Y, theta):
    return Y[0]*np.log(2+theta)+(Y[1]+Y[2])*np.log(1-theta)+Y[3]*np.log(theta)

def calclogM(Y):
    if Y[0]+Y[1]+Y[2] == 0:
        return 0.0
    if 2*(Y[1]+Y[2]) >= Y[0]:
        return Y[0]*np.log(2)
    thetam = (Y[0]-2*(Y[1]+Y[2]))/(Y[0]+Y[1]+Y[2])
    return Y[0]*np.log(2+thetam)+(Y[1]+Y[2])*np.log(1-thetam)


"""
Now we use importance sampling
"""

def imp_samples(Y,maxsamps = -1, maxtime=20, verbose = True):
    samples = np.array([])
    logweights = np.array([])
    timein = time.time()
    timeout = time.time()+maxtime
    cont = True
    while cont:
        X = np.random.uniform()
        samples = np.append(samples, X)
        logweights = np.append(logweights, logpx(Y,X))
        if (maxsamps >0) & (samples.size > maxsamps):
            cont = False
            print('Maximum samples reached')
        if time.time() > timeout:
            cont = False
        
    
    timemid = time.time()
    logweights = logweights - np.max(logweights)
    weights = np.exp(logweights)
    weights = weights / np.sum(weights)
    ESS = np.power(np.linalg.norm(weights), -2)
    if verbose:
        print(samples.size, 'samples were generated with an ESS of', "{0:.2f}".format(ESS))
        print('Additional post-processing time:', "{0:.4f}".format(time.time()-timemid), 'seconds')
    return samples, weights
    
"""
Now we do importance sampling with a smarter proposal distribution
"""


def imp_samples_normal(Y, maxsamps = -1, maxtime = 20, verbose = True):
    samples = np.array([])
    logweights = np.array([])
    timein = time.time()
    timeout = time.time()+maxtime
    cont = True
    
    #calculate MLE:
    zz = np.arange(0.001,0.999,0.001)
    MLE = zz[np.argmax( logpx(Y, zz))]
    var = 1/(Y[0]/((2+MLE)*(2+MLE))+(Y[1]+Y[2])/((1-MLE)*(1-MLE))+Y[3]/(MLE*MLE))
    std = np.sqrt(var)
    
    while cont:
        needsample = True
        while needsample:
            X = np.random.normal(loc=MLE,scale = std)
            if (X > 0) & (X < 1 ):
                needsample = False
            if time.time() > timeout:
                needsample = False
        if (X > 0 ) & (X<1) :
            samples = np.append(samples, X)
            logweights = np.append(logweights, logpx(Y,X)-np.log(norm.pdf(X,loc=MLE,scale=std)))
            if (maxsamps >0) & (samples.size > maxsamps):
                cont = False
                print('Maximum samples reached')
            if time.time() > timeout:
                cont = False
            
    if samples.size > 0:
        timemid = time.time()
        logweights = logweights - np.max(logweights)
        weights = np.exp(logweights)
        weights = weights / np.sum(weights)
        ESS = np.power(np.linalg.norm(weights), -2)
        if verbose:
            print(samples.size, 'samples were generated with an ESS of', "{0:.2f}".format(ESS))
            print('Additional post-processing time:', "{0:.4f}".format(time.time()-timemid), 'seconds')
    else: 
        print('No samples were found!')
    return samples, weights

def MLE(Y):
    zz = np.arange(0.001,0.999,0.001)
    pr = logpx(Y,zz)
    return zz[np.argmax(pr)]

