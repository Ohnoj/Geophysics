# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:37:28 2020

@author: Johno van IJsseldijk

Notes:
-- Rayparameter sampling might be broken at this time!!!!


Translated from MATLAB code: layercac.m
% LAYERCODE 	Compute the  response of 
%		an 1D acoustic medium
% 
% syntax: [T,R]=layercode(cp,rho,dz,nt,dt,np,dp,p0,norm,mul)
%
%
% R	= Reflection response (t,p)
% T	= Transmission response (t,p)
%
% Cp	= Velocity log
% Rho	= Density log
% dz	= depth step
% nt	= number of time samples
% dt	= time sampling
% np	= number op rayparameters
% dp	= rayparameter sampling
% p0	= first rayparameter 
% norm	= 0: flux normalization;  1: pressure normalization
% mul = 0: no multiples;  1: multiples
% nprim = 0: standard scheme, otherwise generate only the primary response of layer nprim

"""

import numpy as npy

def layercac(cp,rho,dz,nt,dt,np,dp,p0,norm=0,mul=0,nprim=0):
    #number of layers
    #------------------
    if len(cp) != len(rho):
    	print('WARNING: discripance between density and velocity log')
    	print(' Smallest is chosen!')
    
    N = min((len(cp),len(rho)))
    
    cp = npy.append(cp,cp[-1])
    rho = npy.append(rho,rho[-1])
    
    #frequencies
    #------------------
    nf      = int((nt/2)+1)
    om      = npy.arange(0,nf)*(2*npy.pi/(nt*dt))
    p     	= npy.arange(0,np)*dp + p0
    
    #initialise the GLOBAL quantities
    #-------------------------
    Rd    =    npy.zeros((nf,np));
    Ru    =    npy.zeros((nf,np));
    Td    =    npy.ones((nf,np));
    Tu    =    npy.ones((nf,np));
    T2    =    npy.ones((nf,np));

    # start the recursion loop over the N-1 layers
    for n in range(N):
        #calculate the local operators
	    #-vertical slowness- (number of layers (N), number of p(np))
        q1 = npy.sqrt(cp[n]**-2-p**2)
        q2 = npy.sqrt(cp[n+1]**-2-p**2)
        
        #-reflection coefficients- and
	    #-flux normalised transmission coefficients-
        
        r     =     (rho[n+1]*q1-rho[n]*q2)/(rho[n+1]*q1+rho[n]*q2);
        if norm == 0:
            td = npy.sqrt(rho[n+1]*q1*rho[n]*q2) / (0.5 * (rho[n+1]*q1+rho[n]*q2))
            tu = td
        else:
            td = 1 + r
            tu = 1 - r
        
        if mul == 0:
            td=1
            tu=1
        
        t2 = tu*td
        
        # Calculated the phase shift operator       
        q = npy.ones((nf,1))*q1
        q = npy.real(q) + 1j*(npy.sign(npy.real(om))*npy.ones((np,1))).T*npy.imag(q)
        r = npy.ones((nf,1))*r
        td = npy.ones((nf,1))*td
        tu = npy.ones((nf,1))*tu
        t2 = npy.ones((nf,1))*t2
        om1 = npy.ones((np,1))*om
        w = npy.exp(1j*om1.T*q*dz)
        w2 = w**2
        
        if mul == 1:
            M = (1-w2*r*Ru)**(-1)
        else:
            M=1
            
        if nprim==0:
            # Calcualted the R downgoing
            Rd = Rd + T2*w2*r*M
        elif n==nprim:
            Rd = Rd + T2*w2*r*M
            
        # Calculate the R upgoing
        Ru = -r + t2*w2*Ru*M
        # Calculate the T downgoing
        Td = td*w*Td*M
        # Calculate the T upgoing
        Tu = tu*w*Tu*M
        # Calculate the T square
        T2 = Td*Tu
        
    Fp = Td**-1
    Fm = Rd*Fp
    
    # Calculated the inverse fft's
    T = npy.concatenate((Td[0:nf-1,:],npy.real([Td[nf-1,:]]),npy.conj(Td[nf-1:1:-1,:])))
    T = npy.real(npy.fft.ifft(npy.conj(T),axis=0))
    
    R = npy.concatenate((Rd[0:nf-1,:],npy.real([Rd[nf-1,:]]),npy.conj(Rd[nf-1:1:-1,:])))
    R = npy.real(npy.fft.ifft(npy.conj(R),axis=0))
    
    F1p = npy.concatenate((Fp[0:nf-1,:],npy.real([Fp[nf-1,:]]),npy.conj(Fp[nf-1:1:-1,:])))
    F1p = npy.real(npy.fft.ifft(npy.conj(F1p),axis=0))
    
    F1m = npy.concatenate((Fm[0:nf-1,:],npy.real([Fp[nf-1,:]]),npy.conj(Fp[nf-1:1:-1,:])))
    F1m = npy.real(npy.fft.ifft(npy.conj(F1m),axis=0)) 
        
    return (T,R,F1p,F1m)


def tdeps(h,nt):
    win = npy.ones((nt,1))
    count=0
    for i in npy.arange(nt-1,int(nt/2),-1):
        if count == 1:
            win[i] = 0
        if abs(h[i]) > 0.0005:
            if count == 0:
                ii=i
                count=1
    if count==1:
        win[ii]=0
        
    return win

def ricker(f0,nt,dt):
    t	=	npy.arange(-round((nt/2)*dt,6),round((nt/2)*dt,6),dt)
    wavsym	= npy.array(1-2*npy.pi**2*f0**2*t**2)*npy.exp(-npy.pi**2*f0**2*t**2)
    wav	=	npy.roll(wavsym,int(nt/2))
    
    return wav


def flatwave(f1,f2,f3,f4,nt,dt):
    freq = npy.fft.rfftfreq(nt,d=dt)
    wav = npy.zeros(freq.shape,dtype=complex)

    wav[npy.logical_and(freq>f1,freq<=f2)]=((npy.cos(npy.pi*freq[npy.logical_and(freq>f1,freq<=f2)]/(f2-f1))+1)/2)[::-1]
    wav[npy.logical_and(freq>f2,freq<f3)]=1
    wav[npy.logical_and(freq>=f3,freq<f4)]=((npy.cos(npy.pi*freq[npy.logical_and(freq>=f3,freq<f4)]/(f4-f3))+1)/2)
    
    return npy.real(npy.fft.irfft(wav))

# nt=2000
# dt=0.002
# freq = npy.fft.rfftfreq(nt,d=dt)
# import matplotlib.pyplot as plt
# plt.figure(5)
# plt.clf()
# #plt.plot(ricker(50,nt,dt))

# plt.plot(flatwave(0,5,80,100,nt,dt))

def conv(A,B,corr=0):
    if corr == 0:
        return npy.real(npy.fft.ifft(npy.fft.fft(A)*npy.fft.fft(B)))
    elif corr == 1:
        return npy.real(npy.fft.ifft(npy.conj(npy.fft.fft(A))*npy.fft.fft(B)))