# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:51:52 2020

@author: Johno van IJsseldijk
"""

import matplotlib.pyplot as plt
import numpy as np
from functions1D import layercac, ricker, flatwave, tdeps, conv
from scipy.ndimage import gaussian_filter1d

# Marchenko iterations
niter= 100

# Modelling parameters
# nt needs to be EVEN
nt	= 32384 #4096*2#16394 #62768 #32384 #16192 #
dt	= 0.002
f0	= 50
t	= np.arange(0,(nt)*dt,dt)
ts	= np.arange(-nt/2*dt,nt/2*dt,dt)
dz	= 200
npar= 1
dp	= 0.0002
p0	= 0
norm= 1
gaus=20

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

Ra = np.reshape(layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,1)[1],nt)

F1p0 = layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,0)[2]
F1p0 = np.roll(np.flipud(np.reshape(F1p0,nt)),1)

#F1p0 = np.real(np.fft.ifft(np.ones_like(F1p0)/np.fft.fft(F1p0)))
Rb = np.reshape(layercac(cpb,rhb,dz,nt,dt,npar,dp,p0,norm,1)[1],nt)
R0b = conv(np.reshape(layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,0)[0],nt),conv(np.reshape(layercac(cpb,rhb,dz,nt,dt,npar,dp,p0,norm,1)[1],nt),np.reshape(layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,0)[0],nt)))

# Model initial estimate focusing function
T,F1p0 = layercac(cpA,rhA,dz,nt,dt,npar,dp,p0,norm,0)[0:3:2]
F1p0 = np.reshape(F1p0,nt)
T = np.reshape(T,nt)

Ttwt = conv(F1p0,F1p0)

# v1pd = np.zeros(nt)
# v1pd[0] = 1
v1pd = conv(wav,conv(F1p0,T))
#v1pd = v1pd/np.max(v1pd)

# Create windowing operators
wina = tdeps(conv(ricker(f0,nt,dt),v1pd),nt)
wina = np.roll(np.reshape(wina,nt),40)
wina = gaussian_filter1d(wina,sigma=10)

winb = tdeps(conv(ricker(f0,nt,dt),Ttwt),nt)
winb = np.roll(np.flipud(np.reshape(winb,nt)),-30)
winb = gaussian_filter1d(winb,sigma=10)

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

# scl = np.max(Uplus)

# Umin *= scl
# Uplus *= scl 

plt.figure(2)
plt.clf()
plt.subplot(222)
plt.plot(ts,np.roll(Umin,int(nt/2)))
plt.axis([0, 3, -.6, 1.2])
plt.title(r'$U^-$')
plt.xlabel('Time [s]')
plt.subplot(221)
plt.plot(ts,np.roll(Uplus,int(nt/2)))
plt.plot(ts,np.roll(wina*Uplus,int(nt/2)))
plt.title(r'$U^+$')
plt.xlabel('Time [s]')
plt.axis([0, 3, -.6, 1.2])
plt.subplot(223)
plt.plot(ts,np.roll(v1p,int(nt/2)))
plt.title(r'$v^+$')
plt.xlabel('Time [s]')
plt.axis([-1.25, 1.25, -.7, 1.2])
plt.subplot(224)
plt.plot(ts,np.roll(v1m,int(nt/2)))
plt.axis([-1.25, 1.25, -.7, .7])
plt.title(r'$v^-$')
plt.xlabel('Time [s]')

Ra_mdd = np.real(np.fft.ifft(np.fft.fft(v1m)/np.fft.fft(v1p)))

Ra_neu = v1m.copy()

VPM = -np.real(np.fft.ifft(np.fft.fft(v1pm)/(np.fft.fft(wav)+1e-2))) #  -v1pm #
print(np.linalg.norm(np.real(np.fft.ifft(np.fft.fft(VPM))),ord=2)**2)

neumann_iter=25
temp = v1m

plt.figure("Ra Neumann = SUM [ (VPM)^k Vmin ]")
plt.clf()
ax1 = plt.subplot(5,5,1)
plt.plot(ts,np.roll(Ra_neu,int(nt/2)))
plt.title("Iter: 0")

for i in range(1,neumann_iter):
    temp = conv(temp,VPM)
    #temp[int(5//dt):] = 0
    if i % 2:
        Ra_neu += temp[:nt]
    else:
        Ra_neu += temp[:nt]
    plt.subplot(5,5,i+1,sharex=ax1,sharey=ax1)
    plt.plot(ts,np.roll(Ra_neu,int(nt/2)))
    plt.plot(ts,np.roll(temp,int(nt/2)),zorder=0)
    plt.title("Iter: {:d}".format(i))
plt.axis([0, 15, -.6, .6])    

plt.figure()
plt.clf()
ax1 = plt.subplot(221)
plt.plot(ts,np.roll(conv(wav,R),int(nt/2)))
plt.axis([0, 3, -.6, 1.2])
plt.title(r'$R_{ab}$ reference')
plt.xlabel('Time [s]')
plt.subplot(222,sharex=ax1,sharey=ax1)
plt.plot(ts,np.roll(conv(wav,Ra),int(nt/2)))
plt.axis([0, 3, -.6, 1.2])
plt.title(r'$R_a$ reference')
plt.xlabel('Time [s]')
plt.subplot(223,sharex=ax1,sharey=ax1)
plt.plot(ts,np.roll(conv(wav,Ra_mdd),int(nt/2)))
plt.axis([0, 3, -.6, 1.2])
plt.title(r'$R_a$ by MDD')
plt.xlabel('Time [s]')
plt.subplot(224,sharex=ax1,sharey=ax1)
plt.plot(ts,np.roll(Ra_neu,int(nt/2)))
#plt.plot(ts,np.roll(v1m,int(nt/2)),zorder=1)
plt.axis([0, 20, -.6, 1.2])
plt.title(r'$R_a$ by Neumann Series')
plt.xlabel('Time [s]')


# plt.figure(4)
# plt.clf()
# plt.plot(ts,np.roll(VPM,int(nt/2)))
# plt.plot(ts,np.roll(v1pm,int(nt/2)))
# plt.xlabel('Time [s]')
# plt.axis([-1.25, 1.25, -.7, 1.2])


# Rb_mdd = np.real(np.fft.ifft(np.fft.fft(Umin)/(np.fft.fft(Uplus))))

# Rb_neu = Umin.copy()

# UPM = -np.real(np.fft.ifft(np.fft.fft(wina*Uplus)/(np.fft.fft(np.abs(wina-1)*Uplus)+.01)))
# # Eq. 37 van der Neut and Wapenaar (2016): http://homepage.tudelft.nl/t4n4v/4_Journals/Geophysics/geo_16b.pdf
# #UPM = np.real(np.fft.ifft(np.fft.fft(wina*Uplus)/(np.fft.fft(wav)+2.5e-7)))
# # Eq. 42 from same paper does not give statisfactory results

# print(np.linalg.norm(np.real(np.fft.ifft(np.fft.fft(UPM))),ord=2)**2)
# # plt.figure(7); plt.clf(); plt.plot(wina*Uplus); plt.plot(-UPM) #plt.plot(wina*Uplus); plt.plot(np.abs(wina-1)*Uplus)

# neumann_iter=25
# temp = Umin.copy()

# plt.figure("Rb Neumann = SUM [ (UPM)^k Umin ]")
# plt.clf()
# ax1 = plt.subplot(5,5,1)
# plt.plot(ts,np.roll(Rb_neu,int(nt/2)))
# plt.title("Iter: 0")

# for i in range(1,neumann_iter):
#     temp = wina*conv(temp,UPM)
#     # temp[int(5//dt):] = 0
#     if i % 2:
#         Rb_neu += temp       
#     else:
#         Rb_neu += temp
#     plt.subplot(5,5,i+1,sharex=ax1,sharey=ax1)
#     plt.plot(ts,np.roll(Rb_neu,int(nt/2)))
#     plt.plot(ts,np.roll(temp,int(nt/2)),zorder=0)
#     plt.title("Iter: {:d}".format(i))
# plt.axis([0, 15, -.6, .6])
# #Rb_neu *= np.min(R0b[:int(3/dt)])/np.min(Rb_neu[:int(3/dt)])

# plt.figure(5)
# plt.clf()
# ax1 = plt.subplot(231)
# plt.plot(ts,np.roll(conv(wav,R),int(nt/2)))
# plt.axis([0, 3, -.6, .6])
# plt.title(r'$R_{ab}$ reference')
# plt.xlabel('Time [s]')
# plt.subplot(232,sharex=ax1,sharey=ax1)
# plt.plot(ts,np.roll(conv(wav,R0b),int(nt/2)))
# plt.axis([0, 3, -.6, .6])
# plt.title(r'$R^0_b$ reference')
# plt.xlabel('Time [s]')
# # plt.subplot(233,sharex=ax1,sharey=ax1)
# # plt.plot(ts,np.roll(conv(wav,Rb),int(nt/2)))
# # plt.axis([0, 3, -.6, .6])
# # plt.title(r'$R_b$ reference')
# # plt.xlabel('Time [s]')
# plt.subplot(234,sharex=ax1,sharey=ax1)
# plt.plot(ts,np.roll(conv(wav,Rb_mdd),int(nt/2)))
# plt.axis([0, 3, -.6, .6])
# plt.title(r'$R^0_b$ by MDD')
# plt.xlabel('Time [s]')
# plt.subplot(235,sharex=ax1,sharey=ax1)
# plt.plot(ts,np.roll(Rb_neu,int(nt/2)))
# plt.plot(ts,np.roll(Umin,int(nt/2)),zorder=1)
# plt.axis([0, 3, -.6, .6])
# plt.title(r'$R^0_b$ by Neumann series')
# plt.xlabel('Time [s]')
# plt.subplot(236,sharex=ax1,sharey=ax1)
# plt.plot(ts,np.roll(np.real(np.fft.ifft(np.fft.fft(conv(v1p,Umin))/(np.fft.fft(wav)+1e-4))),int(nt/2)))
# plt.axis([0, 20, -.6, .6])
# plt.title(r'$R^0_b$ by double focusing')
# plt.xlabel('Time [s]')

# plt.tight_layout()

plt.figure(6)
plt.clf()
depths = int((len(cpB)-1)*dz)
sqvel = np.zeros(depths)
for i in range(len(cpB)):
    sqvel[(i-1)*dz:i*dz] = cpB[i] 
    
plt.plot(sqvel,range(depths))
plt.plot((1000,3250),(int((len(cpA)-1)*dz),int((len(cpA)-1)*dz)),'r--')
plt.ylim((0,depths))
plt.xlim((1000,3250))
plt.gca().invert_yaxis()
plt.xlabel('Velocity [m/s]')
plt.ylabel('Depth [m]')
