# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:51:52 2020

@author: Johno van IJsseldijk
"""

import matplotlib.pyplot as plt
import numpy as np
from functions1D import layercac, ricker, tdeps, conv

# Marchenko iterations
niter= 20

# Modelling parameters
# nt needs to be EVEN
nt	= 32384 
dt	= 0.002
f0	= 20
t	= np.arange(0,(nt)*dt,dt)
ts	= np.arange(-nt/2*dt,nt/2*dt,dt)
dz	= 200
npar= 1
dp	= 0.0002
p0	= 0
norm= 1
gaus=20

wav = ricker(f0,nt,dt) # 

# Make model
cpa = np.array([1500, 1500, 4000, 4000, 1750], dtype=float)
cpb	= np.array([1750, 1750, 1750, 1750], dtype=float)

cpA = cpa 
cpB = np.concatenate((cpa,cpb))
tcut= .8
scl = 1.

rha = cpa
rhb = cpb
rhA = rha
rhB = np.concatenate((rha,rhb))

# Model Reflection Response
[T,R] = layercac(cpB,rhB,dz,nt,dt,npar,dp,p0,norm,1)[:2]
T = np.reshape(T,nt)
R = np.reshape(R,nt)

Ra = np.reshape(layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,1)[1],nt)

F1p0 = layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,0)[2]
F1p0 = np.roll(np.flipud(np.reshape(F1p0,nt)),1)

# Model initial estimate focusing function
T,F1p0 = layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,0)[0:3:2]
F1p0 = np.reshape(F1p0,nt)
T = np.reshape(T,nt)

Ttwt = conv(F1p0,F1p0)

v1pd = conv(wav,conv(F1p0,T))

# Create windowing operators
wina = tdeps(conv(ricker(f0,nt,dt),v1pd),nt)
wina = np.roll(np.reshape(wina,nt),40) # adjust to account for wavelength

winb = tdeps(conv(ricker(f0,nt,dt),Ttwt),nt)
winb = np.roll(np.flipud(np.reshape(winb,nt)),-30) # adjust to account for wavelength

winboth = wina + winb - 1
winboth[winboth < 0] = 0

# Start Marchenko iterations
v1m = winb*conv(R,v1pd)
v1pm  = wina*conv(R,v1m,1)

for i in range(1,niter):
    if i % 2:
        v1m = winb*conv(R,v1pd+v1pm)
    else:
        v1pm  = wina*conv(R,v1m,1)

v1p = v1pd+v1pm   

plt.figure(1)
plt.clf()
plt.subplot(223)
plt.plot(ts,np.roll(v1pd,int(nt/2)))
plt.plot(ts,np.roll(wina,int(nt/2)))
plt.axis([-1, 1, -.6, 1.2])
plt.title('Initial Estimate and window')
plt.xlabel('Time [s]')
plt.subplot(221)
plt.plot(ts,np.roll(np.flipud(conv(ricker(f0,nt,dt),Ttwt)),int(nt/2)))
plt.plot(ts,np.roll(winb,int(nt/2)))
plt.title('Twt time and window')
plt.xlabel('Time [s]')
plt.axis([0, 3, -.6, 1.2])
plt.subplot(222)
plt.plot(ts,np.roll(v1p,int(nt/2)))
plt.plot(ts,np.roll(v1pm,int(nt/2)))
plt.title(r'Downgoing focusing function $v^+$')
plt.xlabel('Time [s]')
plt.axis([-1.25, 1.25, -.7, 1.2])
plt.subplot(224)
plt.plot(ts,np.roll(v1m,int(nt/2)))
plt.axis([-1.25, 1.25, -.7, .7])
plt.title(r'Upgoing focusing function $v^-$')
plt.xlabel('Time [s]')

Umin = conv(R,v1pd+v1pm)-v1m
Uplus = -np.roll(np.flipud(conv(R,v1m,1)-v1p),1)

plt.figure(2)
plt.clf()
plt.subplot(211)
plt.plot(ts,np.roll(v1p,int(nt/2)))
plt.title(r'$v^+$')
plt.xlabel('Time [s]')
plt.axis([-1.25, 1.25, -.7, 1.2])
plt.subplot(212)
plt.plot(ts,np.roll(v1m,int(nt/2)))
plt.axis([-1.25, 1.25, -.7, .7])
plt.title(r'$v^-$')
plt.xlabel('Time [s]')

Ra_mdd = np.real(np.fft.ifft(np.fft.fft(v1m)/np.fft.fft(v1p)))

winc = tdeps(v1pd,nt)
winc = np.roll(np.flipud(np.reshape(winc,nt)),int(tcut/dt))

# Truncate medium, to remove multiples
R_cut = winc*R

plt.figure(3)
plt.clf()
plt.subplot(221)
plt.plot(ts,np.roll(conv(R,wav),int(nt/2)))
plt.plot(ts,np.roll(winboth,int(nt/2)))
plt.axis([0, 3, -.6, 1.2])
plt.title(r'$R$')
plt.xlabel('Time [s]')
plt.subplot(222)
plt.plot(ts,np.roll(conv(Ra_mdd,wav),int(nt/2)))
plt.axis([0, 3, -.6, 1.2])
plt.title(r'$R_a$')
plt.xlabel('Time [s]')

# Start Marchenko iterations
v1m = winb*conv(R_cut,v1pd)
v1pm  = wina*conv(R_cut,v1m,1)

for i in range(1,niter):
    if i % 2:
        v1m = winb*conv(R_cut,v1pd+v1pm)
    else:
        v1pm  = wina*conv(R_cut,v1m,1)

v1p = v1pd+v1pm*scl

Racut_mdd = np.real(np.fft.ifft(np.fft.fft(v1m)/np.fft.fft(v1p)))

plt.figure(2)
plt.subplot(211)
plt.plot(ts,np.roll(v1p,int(nt/2)))
plt.subplot(212)
plt.plot(ts,np.roll(v1m,int(nt/2)))

plt.figure(3)
plt.subplot(223)
plt.plot(ts,np.roll(conv(R_cut,wav),int(nt/2)))
plt.axis([0, 3, -.6, 1.2])
plt.title(r'$R_{cut}$')
plt.xlabel('Time [s]')
plt.subplot(224)
plt.plot(ts,np.roll(conv(Racut_mdd,wav),int(nt/2)))
plt.axis([0, 3, -.6, 1.2])
plt.title(r'$R_{a,cut}$')
plt.xlabel('Time [s]')

plt.figure(4)
plt.clf()
plt.subplot(111)
plt.plot(ts,np.roll(conv(Racut_mdd-Ra_mdd,wav),int(nt/2)))
plt.axis([0, 3, -.01, .01])
plt.plot(ts,np.roll(conv(Racut_mdd-R,wav),int(nt/2)))
plt.axis([0, 3, -.01, .01])
plt.title('Difference')
plt.xlabel('Time [s]')
plt.legend(('MDDs','Actual'))

plt.figure(6)
plt.clf()
depths = int((len(cpB)-1)*dz)
sqvel = np.zeros(depths)
for i in range(len(cpB)):
    sqvel[(i-1)*dz:i*dz] = cpB[i] 
    
plt.plot(sqvel,range(depths))
plt.plot((1000,4500),(int((len(cpA)-1)*dz),int((len(cpA)-1)*dz)),'r--')
plt.ylim((0,depths))
plt.xlim((1000,4500))
plt.gca().invert_yaxis()
plt.xlabel('Velocity [m/s]')
plt.ylabel('Depth [m]')

plt.figure(5,figsize=(4.8,3.2))
plt.clf()
plt.plot(ts,np.roll(conv(R_cut,wav),int(nt/2)))
plt.plot(ts,np.roll(conv(R,wav),int(nt/2)),zorder=0)
twin = (np.sum(winb)-nt/2-30)*dt
plt.plot((twin,twin),(-.6,1.2),'--')
twin = (np.sum(winb)-nt/2+100)*dt
plt.plot((twin,twin),(-.5,.9),'-.')
plt.axis([0, 2, -.5, .9])
plt.title(r'Modelled reflection response',fontsize=15)
plt.xlabel('Time [s]',fontsize=14)
plt.ylabel('Amplitude',fontsize=14)
plt.xticks(ticks=np.linspace(0,2,5),labels=np.linspace(0,2,5),fontsize=14)
plt.yticks(ticks=np.linspace(-.4,.8,4),labels=np.round(np.linspace(-.4,.8,4),1),fontsize=14)  

plt.legend(('Primaries','Multiples','Twt at 1000m','Twt at 1200m'),ncol=1,loc=1,fontsize=12, framealpha=1)
plt.tight_layout()

fd=1000
for ti in (-30,100):
    # Create windowing operators
    wina = tdeps(conv(ricker(f0,nt,dt),v1pd),nt)
    wina = np.roll(np.reshape(wina,nt),40)
    
    winb = tdeps(conv(ricker(f0,nt,dt),Ttwt),nt)
    winb = np.roll(np.flipud(np.reshape(winb,nt)),ti) 

    # Start Marchenko iterations
    v1m = winb*conv(R,v1pd)
    v1pm  = wina*conv(R,v1m,1)
    
    for i in range(1,niter):
        if i % 2:
            v1m = winb*conv(R,v1pd+v1pm)
        else:
            v1pm  = wina*conv(R,v1m,1)
    
    v1p = v1pd+v1pm  
    
    Ra_mdd = np.real(np.fft.ifft(np.fft.fft(v1m)/np.fft.fft(v1p)))
    
    # Start Marchenko iterations
    v1m = winb*conv(R_cut,v1pd)
    v1pm  = wina*conv(R_cut,v1m,1)
    
    for i in range(1,niter):
        if i % 2:
            v1m = winb*conv(R_cut,v1pd+v1pm)
        else:
            v1pm  = wina*conv(R_cut,v1m,1)
    
    v1p = v1pd+v1pm*scl

    Racut_mdd = np.real(np.fft.ifft(np.fft.fft(v1m)/np.fft.fft(v1p)))
    
    plt.figure(100+ti,figsize=(4.8,3.2))
    plt.clf()
    plt.plot(ts,np.roll(conv(Ra_mdd,wav),int(nt/2)))
    plt.plot(ts,np.roll(conv(Racut_mdd,wav),int(nt/2)),'--')
    plt.axis([0, 2, -.5, .9])
    plt.title(r'Retrieved $R_a$ with focal depth at {:d}m'.format(fd),fontsize=15)
    plt.xlabel('Time [s]',fontsize=14)
    plt.ylabel('Amplitude',fontsize=14)
    plt.xticks(ticks=np.linspace(0,2,5),labels=np.linspace(0,2,5),fontsize=14)
    plt.yticks(ticks=np.linspace(-.4,.8,4),labels=np.round(np.linspace(-.4,.8,4),1),fontsize=14)  
    # plt.gca().yaxis.set_ticklabels('')
    plt.legend(('Full response','Primaries only'),fontsize=12)
    plt.tight_layout()
    
    fd=1200