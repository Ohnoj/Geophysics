# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:43:03 2020

@author: Johno van IJsseldijk

"""
#%% Imports and functions

import numpy as np
import numpy.matlib
from matplotlib import cm
from matplotlib import lines
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

#plt.close('all')
if ('fignum' in globals()):
    del fignum

plt.rcParams['savefig.format'] = 'pdf'

import sys

#%% ------------------------- PLOTTING  FUNCTIONS ------------------------- %##

def show_subs(A,i=-1,dt=0.004,acaus=0,titles=None,suptitle=None,figsize=[6.4,4.8],length=5000,perc=100, intmethod = 'hanning',cmap=cm.seismic,name=None,subs=None,scaled=False,ylabel='Time [s]',xlabel='Distance [m]',colorbar=None,eqclim=True):
    if (i > 0):
        fig=plt.figure(i,figsize=figsize)
        plt.clf()
    else:
        fig=plt.figure(figsize=figsize)
        
    if (name != None):
        fig.canvas.set_window_title(name)
    
    mats = len(A)
    if (subs==None):
        subs = (mats//10+1)*10 + (mats % 10)
    
    if (titles==None):
        titles=('',)*mats
    elif (len(titles) < mats):
        titles=titles + ('',)*(mats-len(titles))       
    
    if (mats > (subs//10)*(subs%10)):
        sys.exit('Amount of subplots smaller than amount of matrices')
    if (subs // 100 > 0):
        sys.exit('Maximum subplots size is 9 x 9')
    
    axes=np.array([None]*(subs//10*subs%10))
    
    for k in np.arange(mats):
        if np.all(np.isnan(A[k])):
            continue
        if (k == 0):
            axes[k] = fig.add_subplot(subs//10,subs%10,k+1)
        else:
            axes[k] = fig.add_subplot(subs//10,subs%10,k+1,sharex=axes[0],sharey=axes[0])
        if scaled:
            scale=np.nanmax(np.abs(A))
        else:
            scale=np.max(np.abs(A[k]))    
        im = show_image(A[k],dt=dt,acaus=acaus,i=None, intmethod =intmethod,title=titles[k],length=length,perc=perc,cmap=cmap,scale=scale,ylabel=ylabel,xlabel=xlabel,eqclim=eqclim) # minus to match gray scale of seismic unix
    if (colorbar != None):
        fig.colorbar(im, ax=(axes[axes!=None]).tolist(), format=colorbar)
    if (suptitle != None): plt.suptitle(suptitle,fontsize=16,fontweight='bold')
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()
#    
#    fig.tight_layout()
#    plt.show()
#    plt.tight_layout()
        
def show_image(A,acaus=1,dt=0.004,i=-1,title='',ylabel='Time [s]',xlabel='Distance [m]',intmethod = 'hanning',length=5000,perc=100,cmap=cm.gray,scale=None,eqclim=True):
    if (i == None):
        pass
    elif (i > 0):
        plt.figure(i)
        plt.clf()
    else:
        plt.figure()
    
    if not(scale==None) and not(eqclim):
        pass#print('Only one of scale/eqlim can be used at a time, using matrix clim')
    
    if (scale == None or not(eqclim)):
        vmax=(perc/100)*abs(A).max()
        vmin=(perc/100)*abs(A).min()
    else:
        vmax=(perc/100)*scale
        vmin=-(perc/100)*scale
    
    if eqclim:
        im = plt.imshow(A,cmap=cmap,vmax=vmax, vmin=-vmax,interpolation=intmethod)
    else:
        im = plt.imshow(A,cmap=cmap,vmax=vmax, vmin=vmin,interpolation=intmethod)
        
#    plt.clim(None, 0.5)
    nt = A.shape[0]
    nx = A.shape[1]

    if acaus:
#        tistart = np.floor(nt-np.floor(nt/100)*100)/2
#        tiend = tistart + np.floor(nt/100)*100
#        tdiff = np.floor((np.floor(nt/100)*100)/2)
#        plt.yticks(np.linspace(tistart,tiend,11),np.round(np.linspace(-tdiff*dt,tdiff*dt,11),decimals=1))
        plt.plot([0,nx],[nt//2,nt//2],'k',linewidth=0.5)
    else:
        pass
#        plt.yticks(np.linspace(0,np.floor(nt/100)*100,21),np.round(np.linspace(0,np.floor(nt/100)*100*dt,21),decimals=1))
    
#    plt.xticks(np.linspace(0,nx-1,9),(np.round(np.linspace(-length/2,length/2,9),int(-np.floor(np.log(length))+2)),np.linspace((-length/2),(length/2),9,dtype=int))[length.is_integer() if isinstance(length,float) else True])
        
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.axis('auto')
    plt.xlim([0,nx])
#    plt.ylim([nt-1,0])
    return im

##% ------------------------- Auxilairy FUNCTIONS ------------------------- %##

def Make_Full(A,nrec=1001,t2k=None,nw=513):
    B = np.zeros((nw,nrec,nrec),dtype=complex)
    if t2k is None:
        t2k = np.ones((nw,nrec))
    print(B.shape)
    for i in np.arange(nrec):
        B[:,:,i] = t2k*A[:,i:i+nrec]
    return B

def Make_model(x3,vel,ext=0,length=5000):
    depths = np.arange(0,x3[-1]+1+ext,1)
    
    Mod=np.zeros((depths.size,length))
    
    for i in range(depths.size):
        k = ((depths[i] <= x3).argmax() if np.any(depths[i] <= x3) else -1)
        Mod[i,:] = vel[k]
    
    return Mod


def calc_imps(z1,z2):
    return (z2**2-z1**2)/(z2**2+z1**2)

##% ---------------------------- MDD FUNCTIONS ---------------------------- %%#

def Taper(shp,taplength,axis=1):
    
    mask = np.ones(shp)
    tap = (-np.cos(np.arange(taplength)*np.pi/(taplength))/2+1/2)
    
    if axis == 0:
        mask[np.arange(0,taplength),:] = tap[:,None]
        mask[-np.arange(1,taplength+1),:] = tap[:,None]
    elif axis == 1:
        mask[:,np.arange(0,taplength)] = tap[:,None].T
        mask[:,-np.arange(1,taplength+1)] = tap[:,None].T
    else:
        sys.exit('Only 2 axes can be tapered at this stage')
    
    return mask

def deg2rad(angle):
    return angle*(np.pi/180)

def snell(a1,v1,v2):
    return np.arcsin(v2*np.sin(np.pi/2-a1)/v1)

def Ray(x0,y0,x1,y1,angle=None,color={1:'b',-1:'r'},arrow='->',lw=None,ls='-'):
    if color == None and angle==None:
        sys.exit('Need at least one of angle/color')
    elif color == None:
        color = 'b' if angle >= 0 else 'r'
    
    
#    ax = plt.gca()
    plt.annotate('', xy=(x1,y1), xytext=(x0,y0),
            arrowprops={'arrowstyle': arrow,'fc':color[angle],'ec':color[angle],'ls': ls, 'lw':lw},
            va='center',ha='center', annotation_clip=False)

def Prop(x0,y0,t,angles,depths,v,layeri,updown,intmult=True,FocalDepth=None,Rec=np.inf,c={1:'b',-1:'r'},initial=False,ls='-'):
    if Rec >=0:
        if layeri==len(depths)-2:
            if x0+(t*v[layeri])*np.cos(-angles[layeri]) >= plt.xlim()[1]: 
                if x0 < plt.xlim()[1] and y0+(plt.xlim()[1]-x0)*np.tan(angles[layeri]) < depths[layeri] and updown > 0: 
                    Ray(x0,y0,plt.xlim()[1],y0+(plt.xlim()[1]-x0)*np.tan(angles[layeri]),updown,color=c,ls=ls)
                    return
            
            return
        elif layeri>=len(depths):
            return   
        y1 = y0+(t*v[layeri])*np.sin(angles[layeri])
        x1 = x0+(t*v[layeri])*np.cos(-angles[layeri])

        
        if FocalDepth == None: 
            intmult = intmult
        else:
            intmult = True if y0 > FocalDepth else intmult
        
        ## BOUNDARIES (TOP, BOTTOM, LHS)
        if not(updown==1 and layeri==0):
            if x1 >= plt.xlim()[1] and x0 < plt.xlim()[1] and y0+(plt.xlim()[1]-x0)*np.tan(angles[layeri]) < depths[-1]: 
                if y0+(plt.xlim()[1]-x0)*np.tan(angles[layeri]) < depths[layeri] and updown > 0: 
                    Ray(x0,y0,plt.xlim()[1],y0+(plt.xlim()[1]-x0)*np.tan(angles[layeri]),updown,color=c,ls=ls)
                    return
                elif layeri == 0 and y0 + (plt.xlim()[1]-x0)*np.tan(angles[layeri]) > 0:
                    Ray(x0,y0,plt.xlim()[1],y0+(plt.xlim()[1]-x0)*np.tan(angles[layeri]),updown,color=c,ls=ls)
                    return
                elif (y0+(plt.xlim()[1]-x0)*np.tan(angles[layeri]) > depths[layeri-1] and updown < 0):
                    Ray(x0,y0,plt.xlim()[1],y0+(plt.xlim()[1]-x0)*np.tan(angles[layeri]),updown,color=c,ls=ls)
                    return
    #        return
            
        if y1 < 0 and layeri==0:
            Ray(x0,y0,x0-depths[0]/np.tan(angles[0]),0.1,updown,color=c,ls=ls)
        elif updown==-1 and layeri!=0 and y1 < depths[layeri-1]:
            ynew=depths[layeri-1]-10
            xnew=x0-(depths[layeri]-ynew)/np.tan(angles[layeri]) if not initial else x0-(np.abs(y0-ynew))/np.tan(angles[layeri])
            Ray(x0,y0,xnew,ynew,updown,color=c,ls=ls)
            ynew=depths[layeri-1]
            xnew=x0-(depths[layeri]-ynew)/np.tan(angles[layeri]) if not initial else x0-(np.abs(y0-ynew))/np.tan(angles[layeri])
            if intmult == True:
                Prop(xnew,ynew,t-np.sqrt((xnew-x0)**2+(ynew-y0)**2)/v[layeri],-angles,depths,v,layeri,-updown,True,FocalDepth=FocalDepth,Rec=Rec-1,c=c,ls=ls)
                Prop(xnew,ynew,t-np.sqrt((xnew-x0)**2+(ynew-y0)**2)/v[layeri],angles,depths,v,layeri-1,updown,True,FocalDepth=FocalDepth,Rec=Rec-1,c=c,ls=ls)
            else:
                Prop(xnew,ynew,t-np.sqrt((xnew-x0)**2+(ynew-y0)**2)/v[layeri],angles,depths,v,layeri-1,updown,intmult=intmult,FocalDepth=FocalDepth,Rec=Rec-1,c=c,ls=ls)
            
        elif y1 < depths[layeri]:
            Ray(x0,y0,x1,y1,updown,color=c,ls=ls)
        else:        
    #        (x0,y0) = Ray(100,0,deg2rad(angles[0]),t,3)
            ynew=depths[layeri]
            xnew=x0+(depths[layeri]-depths[layeri-1])/np.tan(angles[layeri]) if not initial else x0+(np.abs(y0-depths[layeri]))/np.tan(angles[layeri])
            
            if Rec == np.inf:
                Prop(xnew,ynew,t-np.sqrt((xnew-x0)**2+(ynew-y0)**2)/v[layeri],angles,depths,v,layeri+1,updown,intmult=intmult,FocalDepth=FocalDepth,Rec=Rec-1,c=c,ls=ls)
                if layeri < len(depths)-2 and v[layeri] != v[layeri+1]:
                    Prop(xnew,ynew,t-np.sqrt((xnew-x0)**2+(ynew-y0)**2)/v[layeri],-angles,depths,v,layeri,-updown,intmult=intmult,FocalDepth=FocalDepth,Rec=Rec-1,c=c,ls=ls)
            elif Rec > 0:
                if layeri < len(depths)-2 and v[layeri] != v[layeri+1]:
                    Prop(xnew,ynew,t-np.sqrt((xnew-x0)**2+(ynew-y0)**2)/v[layeri],-angles,depths,v,layeri,-updown,intmult=intmult,FocalDepth=FocalDepth,Rec=Rec-1,c=c,ls=ls)
                Prop(xnew,ynew,t-np.sqrt((xnew-x0)**2+(ynew-y0)**2)/v[layeri],angles,depths,v,layeri+1,updown,intmult=intmult,FocalDepth=FocalDepth,Rec=Rec-1,c=c,ls=ls)
            ynew=depths[layeri]+10
            xnew=x0+(ynew-depths[layeri-1])/np.tan(angles[layeri]) if not initial else x0+(np.abs(y0-ynew))/np.tan(angles[layeri])    
            Ray(x0,y0,xnew,ynew,updown,color=c,ls=ls)
   
def background(depths,velocities,FocalDepth=1000,i=1):
    plt.imshow((Make_model(depths[:-1],velocities[:-1],00)),cmap=cm.gray_r)
    plt.ylabel('Depth [m]')
    
    plt.clim([1500,4000])
#    plt.yticks(np.linspace(0,1500,16),(np.linspace(0,np.floor(151/1)*1*10-10,16,dtype=int)))
    #cax = plt.colorbar(format='%d')
    #cax.set_label('Velocity [m/s] / Density [kg/m$^3$]')
    plt.plot([0,5000],[FocalDepth,FocalDepth],'w--')
#    plt.xticks(np.linspace(0,5000,11),np.linspace(-2500,2500,11,dtype=int))
     

#    plt.plot(100,0,'k*',markersize=10,clip_on=False)
    
    
    plt.axis('square')
    plt.ylim([1500,0]); plt.xlim([0,4000])
    plt.xticks(np.linspace(0,4000,9),(np.linspace(-2000,2000,9,dtype=int)))
    plt.plot(np.linspace(00,plt.xlim()[1]-00,int(plt.xlim()[1]//100)),np.zeros(int(plt.xlim()[1]//100)),'kv',clip_on=False,ms=5)
    plt.plot(2000,FD,'k*',markersize=15,clip_on=False,markeredgewidth=0.75, markeredgecolor='k')
    
    
def calc_circle_3(xa,ya,xb,yb,xc,yc):
    A = xa*(yb-yc) - ya*(xb-xc) + xb*yc - xc*yb
    B = (xa**2+ya**2)*(yc-yb)+(xb**2+yb**2)*(ya-yc)+(xc**2+yc**2)*(yb-ya)
    C = (xa**2+ya**2)*(xb-xc)+(xb**2+yb**2)*(xc-xa)+(xc**2+yc**2)*(xa-xb)
    
    if (2*A) == 0:
        return (False,False,False)
    else:
        x = -B/(2*A)
        y = -C/(2*A)
        r = np.sqrt((x-xa)**2+(y-ya)**2)
        return (x,y,r)
    
def compute_wavefield(depths,velocities,ox,oz,t,j):    
    i = np.min(np.where(depths > oriz))
    
    t0 = depths[i-1]/velocities[i-1]
    
    xa = ox-velocities[i]*(t-t0)
    za = oz
    xb = ox
    zb = oz+velocities[i]*(t-t0)
    xc = ox+velocities[i]*(t-t0)
    zc = oz
    add_wave(xa,za,xb,zb,xc,zc,depths,velocities,t,i)


def add_wave_only(xa,za,xb,zb,xc,zc,depths,velocities,t,i,primary=1,ud=1,c='r',maxlayers=3):
    global wave
    global angles_rad
    global ax1
    global orix
    global times
    
    xcoords = []
    zcoords = []
    
    (x,z,r) = calc_circle_3(xa,za,xb,zb,xc,zc)
    if np.any((x,z,r)):
        xcoords = r*np.cos(angles_rad) + x
        zcoords = r*np.sin(angles_rad) + z
        xcoords[zcoords < depths[i-1]] = None
        xcoords[zcoords > depths[i]] = None 
    else:
        xcoords = [xa,xb,xc]
        zcoords = [za,zb,zc]
    
    
    
    j=np.min(np.where(wave==None))
    
    wave[j] = ax1.plot(xcoords,zcoords,c)
    
    
def add_wave(xa,za,xb,zb,xc,zc,depths,velocities,t,i,primary=1,ud=1,c='r',maxlayers=3):
    global wave
    global angles_rad
    global ax1
    global orix
    global times
    
    xcoords = []
    zcoords = []
    
    (x,z,r) = calc_circle_3(xa,za,xb,zb,xc,zc)
    if np.any((x,z,r)):
        xcoords = r*np.cos(angles_rad) + x
        zcoords = r*np.sin(angles_rad) + z
        xcoords[zcoords < depths[i-1]] = None
        xcoords[zcoords > depths[i]] = None 
    else:
        xcoords = [xa,xb,xc]
        zcoords = [za,zb,zc]
    
    
    
    j=np.max(np.where(wave==None))
    
    wave[j] = ax1.plot(xcoords,zcoords,c)
    
    if times[j] == 0:
        times[j] = t
    
    # tfirst = (depths[0]-depths[-1]) / velocities[0]
    # tsec = (depths[1] - depths[0]) / velocities[1]
    # (tfirst)*velocities[0]/1e3 + (tsec-tfirst)*velocities[1]/1e3 + (t-tsec)*velocities[2]/1e3
    
    if np.any(zcoords > depths[i]) and (i < maxlayers-1):
        if primary==1:
            r = 0
            t0 = np.zeros((i+2,))
            for k in range(i+1):
                t0[k] = (depths[k]-depths[k-1]) / velocities[k] 
                r += (t0[k]) * velocities[k]
            r += (t-np.sum(t0))*velocities[i+1]
            
            dx = np.sqrt((r)**2 - (depths[i])**2)
            
            
            
            # Primary Transmissions
            add_wave(orix-dx,depths[i],orix,depths[i]+(t-np.sum(t0))*velocities[i+1],orix+dx,depths[i],depths,velocities,t,i+1,primary=primary,ud=ud,c=c,maxlayers=maxlayers)
            
            r = 0
            t0 = np.zeros((i+2,))
            for k in range(i+1):
                t0[k] = (depths[k]-depths[k-1]) / velocities[k] 
                r += (t0[k]) * velocities[k]
            r += (t-np.sum(t0))*velocities[i]
            
            dx = np.sqrt((r)**2 - (depths[i])**2)
            
            # ax1.scatter(orix-dx,depths[i],c='g')
            # if i == 1:
            #     ax1.scatter(orix,depths[i]-(t-np.sum(t0))*velocities[i],c='g')
            
            # Primary Reflections
            add_wave(orix-dx,depths[i],orix,depths[i]-(t-np.sum(t0))*velocities[i],orix+dx,depths[i],depths,velocities,t,i,primary=primary+1,ud=-ud,c='b',maxlayers=maxlayers)  
    if np.all((np.any(zcoords < depths[i-1]),i>0,i<maxlayers-1,primary==2)):
        # Primary transmissions
        add_wave(np.nanmin(np.concatenate((xcoords[:len(angles)//4],xcoords[len(angles)*3//4:]))),          
                 depths[i-1],
                 orix,
                 depths[i-1] - (t - times[j]- (depths[i]-depths[i-1])/velocities[i])*velocities[i-1],
                 np.nanmax(xcoords[len(angles)//4:3*len(angles)//4]),
                 depths[i-1],
                 depths,velocities,t,i-1,primary=primary,ud=ud,c='b',maxlayers=maxlayers)
        
        # FIRST INTERNAL MULTIPLE
        if ud == -1:
            add_wave(np.nanmin(np.concatenate((xcoords[:len(angles)//4],xcoords[len(angles)*3//4:]))),          
                  depths[i-1],
                  orix,
                  depths[i-1] + (t - times[j] - (depths[i]-depths[i-1])/velocities[i])*velocities[i],
                  np.nanmax(xcoords[len(angles)//4:3*len(angles)//4]),
                  depths[i-1],
                  depths,velocities,t,i,primary=primary+1,ud=-ud,c='m',maxlayers=maxlayers)
            
    if np.all((np.any(zcoords > depths[i]),i>=0,i<maxlayers-1,primary>2)): 
        if ud == 1:
            add_wave(np.nanmin(np.concatenate((xcoords[:len(angles)//4],xcoords[len(angles)*3//4:]))),          
                  depths[i],
                  orix,
                  depths[i] - (t - times[j] - (depths[i]-depths[i-1])/velocities[i])*velocities[i],
                  np.nanmax(xcoords[len(angles)//4:3*len(angles)//4]),
                  depths[i],
                  depths,velocities,t,i,primary=primary+1,ud=-ud,c='m',maxlayers=maxlayers)
            
            # if primary>1:
            #     ax1.scatter(np.nanmin(np.concatenate((xcoords[:len(angles)//4],xcoords[len(angles)*3//4:]))),depths[i],c='g')
                
            #     ax1.scatter(orix,
            #               depths[i] + (t - times[j] - (depths[i]-depths[i-1])/velocities[i])*velocities[i+1],c='g')
            #     ax1.scatter(np.nanmax(xcoords[len(angles)//4:3*len(angles)//4]),depths[i],c='g')
                
            add_wave_only(np.nanmin(np.concatenate((xcoords[:len(angles)//4],xcoords[len(angles)*3//4:]))),          
                      depths[i],
                      orix,
                      depths[i] + (t - times[j] - (depths[i]-depths[i-1])/velocities[i])*velocities[i+1],
                      np.nanmax(xcoords[len(angles)//4:3*len(angles)//4]),
                      depths[i],
                      depths,velocities,t,i+1,primary=primary+1,ud=ud,c='m',maxlayers=maxlayers)
            
    if np.all((np.any(zcoords < depths[i-1]),i>0,i<maxlayers-1,primary>2)):
        if ud == -1:
            add_wave(np.nanmin(np.concatenate((xcoords[:len(angles)//4],xcoords[len(angles)*3//4:]))),          
                  depths[i-1],
                  orix,
                  depths[i-1] - (t - times[j] - 2*(depths[i]-depths[i-1])/velocities[i])*velocities[i-1],
                  np.nanmax(xcoords[len(angles)//4:3*len(angles)//4]),
                  depths[i-1],
                  depths,velocities,t,i-1,primary=primary+1,ud=ud,c='m',maxlayers=maxlayers)
           
            add_wave(np.nanmin(np.concatenate((xcoords[:len(angles)//4],xcoords[len(angles)*3//4:]))),          
                  depths[i-1],
                  orix,
                  depths[i-1] + (t - times[j] - 2*(depths[i]-depths[i-1])/velocities[i])*velocities[i],
                  np.nanmax(xcoords[len(angles)//4:3*len(angles)//4]),
                  depths[i-1],
                  depths,velocities,t,i,primary=primary+1,ud=-ud,c='m',maxlayers=maxlayers)
    

     
#%% Actual wavefield modelling

### ACTUAL
depths = np.array(  [0.3,0.9, 1.5,1.5,1.5,0])*1e3 #1.5,1.5,1.51])*1e3#
densities =np.array([1.5  ,2.5 ,3.25, 4.0 ,4.0,4.0])*1e3#3.25,3.25,3.25])*1e3 #
velocities = densities

angles = np.arange(0,360.1,.1)

global angles_rad
angles_rad = deg2rad(angles)
    
FD=0
imno = 1
savedir='./GeneralSeismic/im'

save_to_image=True
global orix
orix=2000
oriz=0
figsize=(10,4)

t0=0
dt=20
tmax=2001

fig = plt.figure(1,figsize=figsize)
plt.clf()
global ax1
ax1=fig.add_subplot(121)
background(depths,velocities,FocalDepth=oriz)
ax1.set_xlabel('Lateral distance [m]')
ax1.xaxis.set_ticks_position('top')
ax1.xaxis.set_label_position('top')

ax2=fig.add_subplot(122)
ax2.scatter(0,0,c='r',s=2,clip_on=False)
xlim=2000
#plt.tight_layout()

global wave
wave=np.empty((100),dtype=lines.Line2D)
global times
times=np.zeros((100))
tfirst=0

for t in np.arange(t0,tmax,dt):
    
    ca1 = np.arcsin(velocities[0]/velocities[1])
    

    t_av1=depths[0]/velocities[0]
    t_av2=(depths[1]-depths[0])/velocities[1]
    t_av3=(depths[2]-depths[1])/velocities[2]
    
    v_av =np.sqrt( (velocities[0]**2*t_av1+velocities[1]**2*t_av2)/(t_av1+t_av2))
    
    r = t*velocities[0]/1e3
    x = r*np.cos(angles_rad) + orix
    z = r*np.sin(angles_rad) + oriz
    x[z>depths[0]] = None
    # wave[0] = ax1.plot(x,z,'r')

    
    if np.any(z>depths[0]):
        if tfirst==0:
            tfirst=t
        r = t*velocities[0]/1e3
        x1 = r*np.cos(angles_rad) + orix
        z1 = r*np.sin(angles_rad) + oriz + depths[0]*2
        x1[z1>depths[0]] = None

        r = (tfirst)*velocities[0]/1e3 + (t-tfirst)*velocities[1]/1e3
        x2 = r*np.cos(angles_rad) + orix
        z2 = r*np.sin(angles_rad) + oriz
        x2[z2<depths[0]] = None
        x2[z2>depths[1]] = None
        
        # HEADWAVE 
        try:
            if (t-tfirst < 120):
                xr = np.unique(x2[(np.round(z2)>depths[0])])[0]
                indexhw = np.where(np.min(np.abs(angles_rad-(1.5*np.pi-ca1)))==np.abs(angles_rad-(1.5*np.pi-ca1)))[0][0]
                indexhw = int(3*len(angles)//4 - round((3*len(angles)//4 - indexhw + 10))*((t-tfirst)/120)) + 10
                
                wave[-1] = ax1.plot([x1[indexhw],xr],[z1[indexhw],depths[0]],'g')
                
                xr = 2*orix-np.unique(x2[(np.round(z2)>depths[0])])[0]
                indexhw = np.where(np.min(np.abs(angles_rad-(1.5*np.pi-ca1)))==np.abs(angles_rad-(1.5*np.pi-ca1)))[0][0]
                indexhw = int(3*len(angles)//4 + round((3*len(angles)//4 - indexhw + 10)*((t-tfirst)/120))) + 10
                wave[-2] = ax1.plot([x1[indexhw],xr],[z1[indexhw],depths[0]],'g')
            else:
                xr = np.unique(x2[(np.round(z2)>depths[0])])[0]
                indexhw = np.where(np.min(np.abs(angles_rad-(1.5*np.pi-ca1)))==np.abs(angles_rad-(1.5*np.pi-ca1)))[0][0]
                wave[-1] = ax1.plot([x1[indexhw],xr],[z1[indexhw],depths[0]],'g')
                xr = 2*orix-np.unique(x2[(np.round(z2)>depths[0])])[0]
                indexhw = np.where(np.min(np.abs(angles_rad-(1.5*np.pi+ca1)))==np.abs(angles_rad-(1.5*np.pi+ca1)))[0][0]
                wave[-2] = ax1.plot([x1[indexhw],xr],[z1[indexhw],depths[0]],'g')
        except:
            pass
                
    compute_wavefield(depths,velocities/1e3,orix,oriz,t,0)
    
    ts = np.arange(0,t,0.01)
    
    twtline = ax2.plot([-xlim,xlim],[t,t],'k')
    ax2.set_xlim([-xlim,xlim]) 
    ax2.set_ylim([tmax,0])
    ax2.set_ylabel('Two-way Traveltime [ms]')
    ax2.set_xlabel('Lateral distance [m]')
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    
    # twt=np.unique(x[(np.round(z)==0)]-orix)
    dirwave=velocities[0]*ts/1e3
    ax2.plot(dirwave,ts,c='r')
    ax2.plot(-dirwave,ts,c='r')
    
    xhw = (ts/1000-2*depths[0]*np.cos(ca1)/velocities[0])*velocities[1]
    
    xhw[xhw < 2*depths[0]*np.tan(ca1)] = None
    ax2.plot(xhw,ts,c='g')
    ax2.plot(-xhw,ts,c='g')
    
    xrefl1 = 2*np.sqrt((((ts/1000)**2*velocities[0]**2/4) - depths[0]**2))
    ax2.plot(-xrefl1,ts,c='b')
    ax2.plot(xrefl1,ts,c='b')
    
    xrefl2=2*np.sqrt((((ts/1000)**2*v_av**2/4)- depths[1]**2))

    ax2.plot(-xrefl2,ts,c='b')
    ax2.plot(xrefl2,ts,c='b')
        
    ca1 = np.arcsin(v_av/velocities[2])   
    
    t_av1=depths[0]/velocities[0]
    t_av2=(2*(depths[1]-depths[0]))/velocities[1]
    t_av3=(depths[2]-depths[1])/velocities[2]
    
    v_av =np.sqrt( (velocities[0]**2*t_av1+velocities[1]**2*t_av2)/(t_av1+t_av2))
    
    
    xrefl3=2*np.sqrt((((ts/1000)**2*v_av**2/4)- (depths[1]+(depths[1]-depths[0]))**2))
    ax2.plot(-xrefl3,ts,c='m')
    ax2.plot(xrefl3,ts,c='m')
        
    t_av1=depths[0]/velocities[0]
    t_av2=(4*(depths[1]-depths[0]))/velocities[1]
    t_av3=(depths[2]-depths[1])/velocities[2]
    
    v_av =np.sqrt( (velocities[0]**2*t_av1+velocities[1]**2*t_av2)/(t_av1+t_av2))-75
    
    xrefl4=2*np.sqrt((((ts/1000)**2*v_av**2/4)- (depths[1]+2*(depths[1]-depths[0]))**2))
    ax2.plot(-xrefl4,ts,c='m')
    ax2.plot(xrefl4,ts,c='m')
    
    txt = ax1.text(350,1425,'t = {:04d}'.format(int(t)),color='w',va='center',ha='center')
    
    plt.tight_layout()
    plt.pause(0.0001) 
    
    if save_to_image:
        plt.savefig(savedir+'{:04d}.png'.format(imno))
        imno += 1
    
    txt.remove()
    
    twtline[0].remove()
    if round(t,2) < tmax-dt:
        ax2.clear()
        for i in range(len(wave)):
            try: 
                wave[i][0].remove()
                wave[i]=None
            except:
                pass

