# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:51:52 2020

@author: Johno van IJsseldijk
"""

import matplotlib.pyplot as plt
import numpy as np
from functions1D import layercac, ricker, flatwave, tdeps, conv

# Marchenko iterations
niter= 20

# Modelling parameters
nt	= 2*8192
dt	= 0.002
f0	= 50
t	= np.arange(0,(nt)*dt,dt)
ts	= np.arange(-nt/2*dt,nt/2*dt,dt)
dz	= 200
npar= 1
dp	= 0.0002
p0	= 0
norm= 1

wav = ricker(f0,nt,dt) # flatwave(0,5,80,100,nt,dt) #

# Make model / medium
cpa = np.array([2000, 1100, 1700, 3000, 2000], dtype=float)
cpb	= np.array([2000, 1200, 2000], dtype=float)
cpA = cpa
cpB = np.concatenate((cpa,cpb))

rha = cpa
rhb = cpb
rhA = rha
rhB = np.concatenate((rha,rhb))

# Model Reflection Response
[T,R] = layercac(cpB,rhB,dz,nt,dt,npar,dp,p0,norm,1)[:2]
T = np.reshape(T,nt)
R = np.reshape(R,nt) #+ 0.01*np.max(R)*(0.5 - np.random.rand(nt))

Rtrunc = np.reshape(layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,1)[:1],nt)

# Model initial estimate focusing function
F1p0 = layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,0)[2]
F1p0 = np.reshape(F1p0,nt)

Tdinv = conv(wav,F1p0)

# Create windowing operator
win = tdeps(conv(ricker(f0,nt,dt),F1p0),nt)
win = np.reshape(win,nt)

# Start Marchenko iterations
f1m = np.flipud(win)*conv(R,Tdinv)
Mp  = win*conv(R,f1m,1)

for i in range(niter):
    f1m = np.flipud(win)*conv(R,Tdinv+Mp)
    Mp  = win*conv(R,f1m,1)

f1p = Tdinv + Mp    

plt.figure(1)
plt.clf()
plt.subplot(223)
plt.plot(ts,np.roll(Tdinv,int(nt/2)))
plt.plot(ts,np.roll(win,int(nt/2)))
plt.axis([-1, 1, -.6, 1.2])
plt.title('Initial Estimate')
plt.xlabel('Time [s]')
plt.subplot(221)
plt.plot(t,conv(wav,R))
plt.plot(t,win)
plt.title('Reflection response')
plt.xlabel('Time [s]')
plt.axis([0, 3, -.6, .4])
plt.subplot(222)
plt.plot(ts,np.roll(f1p,int(nt/2)))
plt.title('Downgoing focusing function')
plt.xlabel('Time [s]')
plt.axis([-1, 1, -.7, 1.2])
plt.subplot(224)
plt.plot(ts,np.roll(f1m,int(nt/2)))
plt.axis([-1, 1, -.7, .7])
plt.title('Upgoing focusing function')
plt.xlabel('Time [s]')

# Find Green's functions between focusing points and surface
Gm = conv(R,f1p) - f1m
Gp = -np.flipud(conv(R,f1m,corr=1) - f1p)

# Retrieve redatumed response from deconvolution of Gm and Gp
R_new = np.real(np.fft.ifft(np.fft.fft(Gm)/np.fft.fft(Gp)))

# Retrieve response without underburden
R_0 = np.real(np.fft.ifft(np.fft.fft(f1m)/np.fft.fft(f1p)))


Radf = conv(Gm,f1p)

# F1p0 = layercac(cpB,rhB,dz,nt,dt,npar,dp,p0,norm,0)[2]
# F1p0 = np.reshape(F1p0,nt)

# Tdinv = np.roll(conv(wav,F1p0),0)

# # Create windowing operator
# win = tdeps(conv(ricker(f0,nt,dt),F1p0),nt)
# win = np.reshape(win,nt)

# # Start Marchenko iterations
# f1m = np.flipud(win)*conv(Radf,Tdinv)
# Mp  = win*conv(Radf,f1m,1)

# for i in range(niter):
#     f1m = np.flipud(win)*conv(Radf,Tdinv+Mp)
#     Mp  = win*conv(Radf,f1m,1)

# f1p = Tdinv + Mp   

# Runder = np.real(np.fft.ifft(np.fft.fft(f1m)/(np.fft.fft(f1p)))) 

plt.figure(2)
plt.clf()
plt.subplot(221)
plt.plot(t,Gm)
plt.axis([0, 3, -.2, .15])
plt.title('Upgoing Green\'s function')
plt.xlabel('Time [s]')
plt.subplot(222)
plt.plot(t,Gp)
plt.axis([0, 3, -.2, .4])
plt.title('Downgoing Green\'s function')
plt.xlabel('Time [s]')
plt.subplot(224)
plt.plot(t,conv(wav,R_new))
plt.axis([0, 3, -.6, .6])
plt.title('Marchenko redatumed response')
plt.xlabel('Time [s]')

# Model of redatumed response
Rb = layercac(cpb,rhb,dz,nt,dt,npar,dp,p0,norm,1)[1]
Rb = np.reshape(Rb,nt)
plt.subplot(223)
plt.plot(t,conv(wav,Rb))
plt.axis([0, 3, -.6, .6])
plt.title('Modelled redatumed response')
plt.xlabel('Time [s]')


plt.figure(3)
plt.clf()
plt.subplot(311)
plt.plot(t,conv(wav,R_0))
plt.axis([0, 3, -.7, .4])
plt.title('Removed underburden (Marchenko)')
plt.xlabel('Time [s]')
plt.subplot(312)
RA = layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,1)[1]
RA = np.reshape(RA,nt)
plt.plot(t,conv(wav,RA))
plt.axis([0, 3, -.7, .4])
plt.title('Removed underburden (Modelled)')
plt.xlabel('Time [s]')
plt.subplot(313)
plt.plot(t,conv(wav,R_0)-conv(wav,RA))
plt.axis([0, 3, -1e-3, 1e-3])
plt.title('Difference')
plt.xlabel('Time [s]')
plt.tight_layout()

plt.figure(4)
plt.clf()

ax1 = plt.subplot(221)
plt.plot(t,Radf)
# plt.plot(t,Runder)

plt.axis([0, 3, -.25, .25])
plt.title('Adaptive double-focusing')
plt.xlabel('Time [s]')
ax2 = plt.subplot(223, sharex=ax1, sharey=ax1)
plt.plot(t,conv(wav,R_new))
plt.axis([0, 3, -.25, .25])
plt.title('MDD redatuming')
plt.xlabel('Time [s]')
ax3 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.plot(t,conv(Tdinv,conv(R,f1p)))
plt.axis([0, 3, -.25, .25])
plt.title('Kees\' f1pd R f1p redatuming')
plt.xlabel('Time [s]')
ax3 = plt.subplot(224, sharex=ax1, sharey=ax1)
plt.plot(t,conv(wav,Rb))
plt.axis([0, 3, -.25, .25])
plt.title('Modelled redatumed response')
plt.xlabel('Time [s]')
# plt.tight_layout()

plt.figure(5); plt.clf()
plt.plot(ts,np.roll(conv(f1p,Rtrunc),int(nt/2)));
plt.axis([-1, 1, -.7, .7])
plt.title('Focusing')

# plt.plot(t,Gm)
# plt.axis([0, 3, -.6, .6])
# plt.title('Gmin')
# plt.xlabel('Time [s]')
# plt.subplot(414)
# plt.plot(ts,np.roll(f1p,int(nt/2)))
# plt.axis([-1, 1, -.7, 1.2])
# plt.title('f1plus')
# plt.xlabel('Time [s]')


